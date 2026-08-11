"""
================================================================================
RUN A QUESTION THROUGH THE PIPELINE
================================================================================

WHAT THIS FILE DOES
-------------------
Asks the graph a question, shows what happened at every step, and -- when the
grounding check is not confident -- pauses so a person can approve or reject the
answer before anyone sees it.

This is the file where the interrupt becomes visible. Everything else in the
graph folder describes the machinery; this runs it.

HOW THE PAUSE ACTUALLY WORKS
----------------------------
This is the part worth reading slowly, because it does not work the way most
people assume.

When you invoke the graph, one of two things happens:

    The answer was grounded
        Every node runs, final_answer is filled in, and invoke() returns a
        finished result. Nothing pauses.

    The answer needed review
        The graph runs as far as the output guardrail, then STOPS -- because we
        compiled it with interrupt_before=["human_review"].

        invoke() RETURNS at that point. It does not block, sleep, or wait. The
        function call finishes and control comes back here, with the state saved
        by the checkpointer.

So a pause is not something happening inside the graph. It is the graph handing
control back to us, mid-run, and waiting to be told to continue.

RESUMING
--------
To continue, we do two things:

    1. update_state()  write the reviewer's decision into the saved state
    2. invoke(None)    carry on from where we stopped

The None is the important detail. Passing None means "do not start a new run --
resume the existing one". Passing a state instead would begin again from the top.

WHY THIS DESIGN IS WORTH THE TROUBLE
------------------------------------
Because the wait costs nothing while it lasts.

Nothing is frozen inside a running function. The state sits in the checkpointer,
and the program is free. A wait of one second and a wait of three days are the
same thing to the graph.

That is what makes this pattern work in a real system, where the reviewer is a
person on another continent looking at a queue -- not somebody sitting at your
terminal.

ABOUT thread_id
---------------
Every run needs an identifier. The checkpointer uses it to know which saved state
to resume when a run continues.

Here we make one up per question. In a real system it would be something durable
-- a request ID, a ticket number -- so a paused run can be found and resumed
later by a different process entirely.

HOW TO RUN
----------
    python run_query.py

    python run_query.py "How did the risk disclosures change?"

    python run_query.py "How did revenue change?" Eternal

Ingestion must have been run first.

================================================================================
"""

import sys
import uuid

import config
from graph.build_graph import build_graph
from graph.state import create_initial_state


# ==============================================================================
# SECTION 1 - PRINTING WHAT HAPPENED
# ==============================================================================
# These helpers exist only to make the run readable. None of them affect what the
# graph does.
#
# They earn their place because the whole point of this file is showing the
# pipeline working. A run that printed only the final answer would hide every
# step we spent the project building.
# ==============================================================================

def print_heading(text: str) -> None:
    """Print a section heading."""
    print()
    print("=" * 74)
    print(text)
    print("=" * 74)


def show_retrieval(state: dict) -> None:
    """
    Show what retrieval and reranking did.

    The interesting column is the movement: which chunks the reranker promoted
    out of the candidate pile, and how many it discarded. Without the before, the
    after says nothing.
    """
    candidates = state.get("candidate_chunks") or {}
    reranked = state.get("reranked_chunks") or {}

    if not reranked:
        return

    print_heading("RETRIEVAL")

    for fiscal_year in sorted(reranked.keys()):
        kept = reranked[fiscal_year]
        found = candidates.get(fiscal_year, [])

        print()
        print(f"  {fiscal_year} - {len(found)} candidates found, "
              f"{len(kept)} kept after reranking")
        print()

        # Work out where each surviving chunk sat before reranking, so the
        # movement is visible. We match on the text, which is unique per chunk.
        original_rank = {r["text"]: i + 1 for i, r in enumerate(found)}

        for rank, result in enumerate(kept, start=1):
            preview = result["text"].replace("\n", " ")[:46]
            was = original_rank.get(result["text"], "?")
            print(f"    {rank}. score {result.get('rerank_score', 0):>7.3f}  "
                  f"(was #{was})  {preview}...")


def show_grounding(state: dict) -> None:
    """Show what the output guardrail concluded."""
    verdict = state.get("grounding_verdict")

    if verdict is None:
        return

    print_heading("GROUNDING CHECK")
    print()
    print(f"  verdict   {verdict}")
    print(f"  score     {state.get('grounding_score', 0):.2f}")

    unsupported = state.get("unsupported_items") or []
    if unsupported:
        print(f"  flagged   {unsupported[0][:60]}")


# ==============================================================================
# SECTION 2 - THE HUMAN REVIEW PROMPT
# ==============================================================================

def ask_reviewer(state: dict) -> dict:
    """
    Show the reviewer what they need, and collect their decision.

    Returns the fields to write back into the state before resuming.

    WHAT A REVIEWER ACTUALLY NEEDS
    ------------------------------
    Three things, and leaving any of them out makes review slow enough that
    people stop doing it properly:

      1. The answer, in full.
      2. What specifically was flagged -- not "something is unsupported" but the
         exact claim or figure that could not be traced.
      3. The extracts it was supposed to come from, so they can check without
         opening a 300-page PDF.

    A reviewer given only "this answer may be ungrounded, approve or reject?"
    will start approving on reflex within the hour. That looks like oversight
    while providing none, which is worse than having no review step at all.
    """
    print_heading("HUMAN REVIEW REQUIRED")

    print()
    print("  The graph has paused. Nothing has been shown to the user.")
    print()
    print("-" * 74)
    print("  DRAFT ANSWER")
    print("-" * 74)
    print()
    print(state["draft_answer"])

    unsupported = state.get("unsupported_items") or []
    if unsupported:
        print()
        print("-" * 74)
        print("  WHY IT WAS FLAGGED")
        print("-" * 74)
        print()
        for item in unsupported:
            print(f"  - {item}")

    print()
    print("-" * 74)
    print("  THE EXTRACTS THIS SHOULD HAVE COME FROM")
    print("-" * 74)

    for fiscal_year, results in sorted(state["reranked_chunks"].items()):
        print()
        print(f"  {fiscal_year}")
        for result in results[:3]:
            preview = result["text"].replace("\n", " ")[:64]
            print(f"    - {preview}...")

    # ---------------------------------------------------------------------
    # Collect the decision
    # ---------------------------------------------------------------------
    print()
    print("-" * 74)
    print()

    decision = ""
    # Keep asking until we get something valid. Accepting an unrecognised answer
    # and guessing what it meant would be a poor habit in a review step.
    while decision not in ("a", "r"):
        decision = input("  approve or reject? [a/r]: ").strip().lower()

    if decision == "r":
        return {"review_decision": "rejected"}

    # An approving reviewer may want to correct the wording. Pressing enter keeps
    # the draft as it is.
    note = input("  edited answer (or press enter to approve as written): ").strip()

    return {
        "review_decision": "approved",
        "review_note": note or None,
    }


# ==============================================================================
# SECTION 3 - RUNNING ONE QUESTION
# ==============================================================================

def run(question: str, company: str) -> None:
    """
    Put one question through the graph, pausing for review if needed.
    """
    graph = build_graph()

    # Every run needs an identifier so the checkpointer knows which saved state
    # belongs to it. uuid4 generates a random one.
    #
    # This config dictionary is LangGraph's, and is unrelated to our config.py --
    # an unfortunate name collision worth noticing so the two are not confused.
    thread = {"configurable": {"thread_id": str(uuid.uuid4())}}

    print_heading("QUESTION")
    print()
    print(f"  {question}")
    print(f"  ({company})")

    # ---------------------------------------------------------------------
    # Start the run
    # ---------------------------------------------------------------------
    # This returns either a finished result, or the state as it stood when the
    # graph paused. We find out which by looking at what came back.
    state = graph.invoke(
        create_initial_state(question=question, company=company),
        thread,
    )

    # ---------------------------------------------------------------------
    # Was it blocked at the input?
    # ---------------------------------------------------------------------
    if not state["input_allowed"]:
        print_heading("BLOCKED BY THE INPUT GUARDRAIL")
        print()
        print(f"  reason (internal): {state['input_block_reason']}")
        print()
        print("  Nothing was retrieved and no answer was generated. The question")
        print("  cost one embedding call and stopped there.")
        print()
        print("-" * 74)
        print(f"  {state['final_answer']}")
        print()
        return

    show_retrieval(state)
    show_grounding(state)

    # ---------------------------------------------------------------------
    # Did the graph pause?
    # ---------------------------------------------------------------------
    # We ask the graph where it currently is. If a node is waiting to run, the
    # run is paused; if nothing is next, it finished.
    #
    # .next holds the nodes queued to run when the graph resumes. An empty .next
    # means there is nothing left to do.
    snapshot = graph.get_state(thread)

    if snapshot.next:
        # ------------------------------------------------------------------
        # Paused. Collect a decision and resume.
        # ------------------------------------------------------------------
        review = ask_reviewer(state)

        # Write the decision into the saved state. This does not run anything --
        # it only updates what is stored, so the human_review node will find the
        # decision waiting when it eventually runs.
        graph.update_state(thread, review)

        # Resume. The None is what makes this a continuation rather than a new
        # run -- passing a state here would start again from the beginning.
        state = graph.invoke(None, thread)

        print_heading("RESUMED")
        print()
        print(f"  reviewer decision: {state['review_decision']}")

    # ---------------------------------------------------------------------
    # Done either way
    # ---------------------------------------------------------------------
    print_heading("ANSWER SHOWN TO THE USER")
    print()
    print(state["final_answer"])
    print()

    print("-" * 74)
    print(f"  reviewed by a human: {state['review_required']}")
    print()


# ==============================================================================
# SECTION 4 - COMMAND LINE
# ==============================================================================

# The default question is a comparison, deliberately. A single-year lookup would
# exercise retrieval and generation but would say nothing about the thing this
# project is built for.
DEFAULT_QUESTION = "How did the company's disclosure of currency risk change between the two years?"
DEFAULT_COMPANY = "Infosys"


if __name__ == "__main__":
    # sys.argv holds the words typed after the command. argv[0] is the script
    # name itself, so the question is argv[1] and the company argv[2].
    #
    # len(sys.argv) > 1 checks a word was actually supplied before reading it --
    # without that check, running with no arguments would raise an IndexError.
    question = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_QUESTION
    company = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_COMPANY

    if company not in config.COMPANIES:
        print(f"Unknown company: {company}")
        print(f"Available: {', '.join(config.COMPANIES)}")
        sys.exit(1)

    run(question, company)