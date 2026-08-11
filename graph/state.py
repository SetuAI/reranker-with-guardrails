"""
================================================================================
GRAPH STEP 1 - THE STATE
================================================================================

WHAT THIS FILE DOES
-------------------
Defines the single object that travels through the pipeline, collecting
information at every step.

There is no logic here. It is a description of a shape. But it is worth
understanding properly before reading anything else in the graph folder, because
every other file works by reading from and writing to this one object.

WHAT LANGGRAPH IS, IN ONE PARAGRAPH
-----------------------------------
So far the project has been ordinary functions calling other functions. LangGraph
replaces that with a set of NODES connected by EDGES.

Each node is a function that does one job. Edges decide which node runs next --
and crucially, that decision can depend on what has happened so far. That is what
lets the pipeline branch: block a question here, send an answer for human review
there, skip straight to the end in another case.

HOW NODES TALK TO EACH OTHER
----------------------------
They do not call each other, and they do not pass arguments around. Nodes never
know what runs before or after them.

Instead there is one shared state object. Every node receives it, reads whatever
it needs, and returns the fields it wants to change. LangGraph merges those
changes in and hands the updated state to the next node.

    state -> [node] -> updated state -> [node] -> updated state -> ...

That indirection is what makes the graph rearrangeable. Adding a guardrail
between two steps means adding a node, not editing the two nodes on either side.

WHY THE STATE LIVES IN ITS OWN FILE
-----------------------------------
Because it is the contract between every node in the project. When a node reads
state["reranked_chunks"], that field has to exist and mean what the node expects.

Keeping the whole contract in one short file means you can see everything a node
might read or write on one screen. Scattered across the node definitions, you
would have to read all of them to know what the state contains.

================================================================================
"""

from typing import Dict, List, Optional, TypedDict


# ==============================================================================
# THE STATE
# ==============================================================================
# ABOUT TypedDict
# ---------------
# A TypedDict is an ordinary Python dictionary, with a written description of
# which keys it should have and what type each value should be.
#
# The important word is "should". Nothing is enforced at runtime. TypedDict does
# not check anything, and putting a wrong value in raises no error.
#
# So what is it for? Two things, both about the people reading the code:
#
#   1. Documentation that cannot drift. A comment describing the state would go
#      stale the moment someone adds a field. This is the actual definition, so
#      it stays correct.
#
#   2. Editor support. Because the keys are declared, your editor can offer them
#      as you type state["..."] and warn you about typos. Without it,
#      state["reranked_chunk"] instead of state["reranked_chunks"] is a mistake
#      you find at runtime, if at all.
#
# LangGraph also uses this class to know the shape of the state it is passing
# around, which is why the graph is built with GraphState rather than a plain
# dictionary.
#
# ABOUT Optional
# --------------
# Optional[str] means "a string, or None".
#
# Almost every field here is Optional, and that is not laziness. When the graph
# starts, only the question has been filled in -- everything else is genuinely
# empty and gets filled in as the run progresses. A field being None is
# meaningful information: it means the step that would have populated it has not
# run yet, or was skipped.
# ==============================================================================

class GraphState(TypedDict):
    """
    Everything that flows through the pipeline.

    The fields are grouped below in the order they get filled in, so reading this
    class top to bottom is roughly the same as watching a question travel through
    the graph.
    """

    # --------------------------------------------------------------------------
    # SET AT THE START - what the user asked
    # --------------------------------------------------------------------------

    # The question, exactly as typed. Never modified, so it is always available
    # for the guardrails and for the final answer.
    question: str

    # Which company the question is about. In this project it is supplied along
    # with the question rather than being detected from it -- a deliberate
    # simplification, since working out the company from the wording is a
    # separate problem that would distract from the pipeline itself.
    company: str

    # --------------------------------------------------------------------------
    # SET BY THE INPUT GUARDRAILS
    # --------------------------------------------------------------------------

    # Whether the question survived the input checks. When this is False, the
    # graph skips retrieval and generation entirely and goes straight to the end.
    input_allowed: bool

    # Why it was blocked, for the logs. None when nothing was blocked.
    input_block_reason: Optional[str]

    # --------------------------------------------------------------------------
    # SET BY RETRIEVAL AND RERANKING
    # --------------------------------------------------------------------------

    # The wide candidate set from vector search, kept per year.
    #
    #     {"FY2025": [...20 chunks...], "FY2026": [...20 chunks...]}
    #
    # WHY KEEP THIS AFTER RERANKING HAS FINISHED WITH IT
    # --------------------------------------------------
    # Nothing downstream reads it. We keep it because it is the only way to see
    # what reranking actually did -- which chunks it promoted and which it threw
    # away. Without the before, the after tells you nothing.
    #
    # In a production system you might drop this to save memory. Here, being able
    # to inspect the pipeline is the point.
    candidate_chunks: Optional[Dict[str, List[Dict]]]

    # The survivors, per year, after reranking.
    #
    #     {"FY2025": [...5 chunks...], "FY2026": [...5 chunks...]}
    #
    # This is what the model actually sees.
    reranked_chunks: Optional[Dict[str, List[Dict]]]

    # --------------------------------------------------------------------------
    # SET BY GENERATION
    # --------------------------------------------------------------------------

    # The answer the model wrote. This is a draft, not something the user has
    # seen -- the output guardrails still have to pass it.
    draft_answer: Optional[str]

    # --------------------------------------------------------------------------
    # SET BY THE OUTPUT GUARDRAILS
    # --------------------------------------------------------------------------

    # The verdict from the factual check: "grounded", "ungrounded", or
    # "needs_review". This one field decides whether the graph goes straight to
    # the user or stops and waits for a person.
    grounding_verdict: Optional[str]

    # What proportion of the numbers in the answer were found in the retrieved
    # chunks. Kept separately from the verdict because the reviewer wants the
    # evidence, not just the conclusion.
    grounding_score: Optional[float]

    # Specific claims or numbers that could not be traced back to the sources.
    #
    # This is the single most useful field for a human reviewer. "Something is
    # unsupported" gives them nothing to work with; "the figure 15.6% does not
    # appear in any retrieved chunk" tells them exactly where to look.
    unsupported_items: Optional[List[str]]

    # --------------------------------------------------------------------------
    # SET BY THE HUMAN REVIEW STEP
    # --------------------------------------------------------------------------

    # Whether the graph paused for a person. Useful for reporting afterwards --
    # what proportion of questions needed review is the number that tells you
    # whether your thresholds are set sensibly.
    review_required: bool

    # What the reviewer decided: "approved", "rejected", or None if no review
    # happened.
    review_decision: Optional[str]

    # Anything the reviewer typed. If they corrected the answer, the corrected
    # text lands here.
    review_note: Optional[str]

    # --------------------------------------------------------------------------
    # THE END
    # --------------------------------------------------------------------------

    # What the user actually sees.
    #
    # WHY THIS IS SEPARATE FROM draft_answer
    # --------------------------------------
    # Keeping them apart means the graph always records both what the model
    # produced AND what was shown, and the two are frequently different: a
    # blocked question yields a refusal, a rejected review yields a notice, an
    # approved review may yield the reviewer's edited text.
    #
    # If generation wrote straight into one field, that history would be
    # overwritten and lost. Which is precisely the record you want when someone
    # asks why a particular user saw a particular thing.
    final_answer: Optional[str]


# ==============================================================================
# CREATING A FRESH STATE
# ==============================================================================

def create_initial_state(question: str, company: str) -> GraphState:
    """
    Build the starting state for a new question.

    WHY EVERY FIELD IS SET EXPLICITLY, INCLUDING THE EMPTY ONES
    -----------------------------------------------------------
    We could pass only the question and company and let the rest appear as nodes
    fill them in. Python dictionaries would allow it.

    Setting them all to None upfront is better for one practical reason: reading
    a key that does not exist raises a KeyError and stops the graph, whereas
    reading a key set to None simply gives back None.

    That turns a whole category of crash into a value a node can check for. A
    node that runs before the field it expects has been populated can handle it
    calmly instead of bringing the run down.

    It also means this function doubles as a complete inventory of the state --
    everything the pipeline can hold, in one place.
    """
    return GraphState(
        question=question,
        company=company,

        # Input guardrails -- we start optimistic and let the checks say
        # otherwise. Starting at False would mean a missing guardrail node
        # silently blocks everything.
        input_allowed=True,
        input_block_reason=None,

        # Retrieval
        candidate_chunks=None,
        reranked_chunks=None,

        # Generation
        draft_answer=None,

        # Output guardrails
        grounding_verdict=None,
        grounding_score=None,
        unsupported_items=None,

        # Human review
        review_required=False,
        review_decision=None,
        review_note=None,

        # Final
        final_answer=None,
    )


# ==============================================================================
# RUNNING THIS FILE DIRECTLY
# ==============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("GRAPH STATE")
    print("=" * 70)
    print()

    state = create_initial_state(
        question="How did the company's currency risk disclosure change?",
        company="Infosys",
    )

    print("  A fresh state, before any node has run:")
    print()

    for key, value in state.items():
        # repr() shows None as the word None rather than an empty space, which
        # makes the emptiness visible rather than looking like a formatting bug.
        print(f"    {key:<22} {value!r}")

    print()
    print("  Only question, company and the two defaults are filled in.")
    print("  Everything else is None, waiting for the node that populates it.")
    print()