
"""
================================================================================
GRAPH STEP 2 - THE NODES
================================================================================

WHAT THIS FILE DOES
-------------------
Defines every step of the pipeline as a separate function.

A node is just a function with a very particular shape:

    def some_node(state: GraphState) -> dict:
        ...
        return {"field_i_changed": new_value}

It receives the whole state, reads whatever it needs, does one job, and returns
ONLY the fields it wants to change. LangGraph merges those changes into the state
and passes the result to whichever node comes next.

TWO RULES THAT MAKE NODES WORK
------------------------------
1. A node never calls another node. It does not know what ran before it or what
   runs after. All communication happens through the state.

2. A node returns changes, not the whole state. Returning {"draft_answer": "..."}
   updates that one field and leaves everything else untouched. You do not need
   to copy forward the fields you did not touch.

Together these are what make the pipeline rearrangeable. Inserting a guardrail
between two steps means writing a new node -- not editing the two on either side.

WHAT IS NOT IN THIS FILE
------------------------
The order. Nothing here says which node runs next; every function is written as
though it might run at any time.

That ordering lives entirely in build_graph.py, and keeping it out of here is the
whole point of the separation.

================================================================================
"""

import sys
from pathlib import Path
from typing import Dict, List

from openai import OpenAI

sys.path.append(str(Path(__file__).parent.parent))

import config
from graph.state import GraphState
from guardrails.topic import check_topic, REFUSAL_MESSAGE, BLOCK
from guardrails.factual import (
    check_grounding,
    collect_source_texts,
    GROUNDED,
    NEEDS_REVIEW,
)
from retrieval.search import search_both_years
from retrieval.rerank import rerank


client = OpenAI(api_key=config.OPENAI_API_KEY)


# ==============================================================================
# NODE 1 - INPUT GUARDRAIL
# ==============================================================================

def input_guardrail_node(state: GraphState) -> dict:
    """
    Check the question before anything expensive happens.

    WHY THIS RUNS FIRST, BEFORE RETRIEVAL
    -------------------------------------
    Position in the graph is a design decision, not an accident.

    A blocked question costs one embedding call here. The same question allowed
    through would cost two searches, ten reranker passes, one large model call to
    generate, and another to check grounding.

    Putting the cheapest rejection at the very front means the questions we do
    not want to answer are also the questions that cost us almost nothing.

    That is the general shape of guardrail placement: reject early, and the
    ordering pays for itself.
    """
    decision = check_topic(state["question"])

    if decision.verdict == BLOCK:
        return {
            "input_allowed": False,
            "input_block_reason": decision.reason,
            # We write the refusal straight into final_answer. That way the graph
            # can jump directly to the end without any later node needing to know
            # that a block happened.
            "final_answer": REFUSAL_MESSAGE,
        }

    return {"input_allowed": True}


# ==============================================================================
# NODE 2 - RETRIEVAL
# ==============================================================================

def retrieve_node(state: GraphState) -> dict:
    """
    Search each financial year separately for chunks relevant to the question.

    Returns the WIDE candidate set -- around 20 chunks per year -- not the final
    selection. Narrowing happens in the next node.

    Splitting search and reranking into two nodes rather than one is deliberate.
    It means the state ends up holding both the before and the after, so you can
    see exactly what reranking changed. If they were one node, the candidates
    would be discarded inside it and the reranker's contribution would be
    invisible.
    """
    candidates = search_both_years(
        question=state["question"],
        company=state["company"],
    )

    return {"candidate_chunks": candidates}


# ==============================================================================
# NODE 3 - RERANKING
# ==============================================================================

def rerank_node(state: GraphState) -> dict:
    """
    Cut each year's candidates down to the few that genuinely answer the question.

    WHY EACH YEAR IS RERANKED SEPARATELY
    ------------------------------------
    We could pool both years' candidates and rerank them together. We do not, and
    the reason matters.

    A combined ranking would let one year dominate the final selection. On this
    corpus that is a real risk, because the two reports describe the same topics
    in nearly identical language -- so the top five could easily come almost
    entirely from whichever year happened to phrase things marginally closer to
    the question.

    The result would be a "comparison" built on evidence from one side, and
    nothing in the output would reveal the imbalance.

    Reranking within each year guarantees a fair number of strong chunks from
    both. Only then is there something real to compare.
    """
    reranked = {}

    for fiscal_year, candidates in state["candidate_chunks"].items():
        reranked[fiscal_year] = rerank(state["question"], candidates)

    return {"reranked_chunks": reranked}


# ==============================================================================
# NODE 4 - GENERATION
# ==============================================================================
# The prompt below is where the comparison actually happens, and three things in
# it are doing real work.
#
#   1. The two years are presented as clearly SEPARATE blocks, each labelled.
#      Pooled into one undifferentiated pile of text, the model would have no
#      reliable way to tell which statement came from which year -- the reports
#      read almost identically -- and it would guess.
#
#   2. The model is told to say when the sources do not show a change. Without
#      that permission, a model asked "what changed?" will find a change,
#      because the question implies one exists. That is one of the most common
#      ways hallucinations get invited by the prompt itself.
#
#   3. The model is told not to give investment advice. The topic guardrail
#      already blocks such questions at the input, but a question can be
#      perfectly in scope and still tempt an advisory answer -- "margins
#      declined" can easily slide into "which is concerning for investors".
#      Guardrails and prompts are layers, not alternatives.
# ==============================================================================

GENERATION_PROMPT = """You answer questions about {company}'s annual reports by \
comparing two financial years.

Use ONLY the extracts below. If they do not contain enough information to answer,
say so plainly rather than filling the gap from general knowledge.

If the extracts show no meaningful change between the two years, say that. Do not
manufacture a change because the question implies one.

When you state a figure, it must appear in the extracts. Do not calculate,
estimate or round beyond what is written.

Do not give investment advice or opinions on whether the company is a good
investment. Describe what the reports say.

Structure your answer as:
- what FY2025 said
- what FY2026 said
- what changed, or that nothing meaningful changed

=== FY2025 EXTRACTS ===
{fy2025_extracts}

=== FY2026 EXTRACTS ===
{fy2026_extracts}

QUESTION: {question}"""


def format_extracts(results: List[Dict]) -> str:
    """
    Turn a year's chunks into numbered text for the prompt.

    The numbering is not decoration. It gives the model a way to refer to a
    specific extract, and it gives us a way to check afterwards which extract an
    answer leaned on.

    If a year produced no chunks, we say so explicitly rather than leaving the
    section blank. An empty section is ambiguous -- the model cannot tell whether
    nothing was found or whether something went wrong -- and it may quietly fill
    the gap itself.
    """
    if not results:
        return "(no relevant extracts found for this year)"

    parts = []
    for number, result in enumerate(results, start=1):
        parts.append(f"[{number}] {result['text']}")

    return "\n\n".join(parts)


def generate_node(state: GraphState) -> dict:
    """
    Write the comparison answer from the retrieved extracts.

    Temperature is 0.0 so the same question and the same extracts always produce
    the same answer.

    That matters more than usual here. The next node checks this answer for
    grounding, and if generation varied between runs, an answer could pass the
    check one moment and fail it the next with nothing having changed. A pipeline
    whose behaviour is not reproducible cannot be tested or debugged.
    """
    reranked = state["reranked_chunks"]

    prompt = GENERATION_PROMPT.format(
        company=state["company"],
        question=state["question"],
        # .get with a default of [] means a missing year produces the "no
        # extracts" message rather than a KeyError that stops the graph.
        fy2025_extracts=format_extracts(reranked.get("FY2025", [])),
        fy2026_extracts=format_extracts(reranked.get("FY2026", [])),
    )

    response = client.chat.completions.create(
        model=config.GENERATION_MODEL,
        temperature=0.0,
        messages=[{"role": "user", "content": prompt}],
    )

    return {"draft_answer": response.choices[0].message.content.strip()}


# ==============================================================================
# NODE 5 - OUTPUT GUARDRAIL
# ==============================================================================

def output_guardrail_node(state: GraphState) -> dict:
    """
    Check the draft answer against the extracts it was supposed to come from.

    This node does not decide what happens next. It records a verdict, and
    build_graph.py reads that verdict to choose the route.

    Keeping those apart is worth noticing. A node that both judged AND routed
    would be doing two jobs, and changing the routing rule would mean editing the
    checking logic. As written, the routing rule lives in one place in
    build_graph.py and this node never needs to change.
    """
    sources = collect_source_texts(state["reranked_chunks"])
    decision = check_grounding(state["draft_answer"], sources)

    return {
        "grounding_verdict": decision.verdict,
        "grounding_score": decision.score,
        "unsupported_items": decision.unsupported_items,
        # Whether a person is needed. build_graph.py routes on this.
        "review_required": decision.verdict != GROUNDED,
    }


# ==============================================================================
# NODE 6 - HUMAN REVIEW
# ==============================================================================

def human_review_node(state: GraphState) -> dict:
    """
    Handle a reviewer's decision.

    THIS NODE DOES NOT DO THE WAITING
    ---------------------------------
    That is the part people expect and it is worth being clear about.

    The pause is not created here. It is created by LangGraph itself, configured
    in build_graph.py to interrupt BEFORE this node runs. When the graph reaches
    that point it simply stops and hands control back to whoever called it.

    So this function only runs AFTER a human has already responded and the graph
    has been told to resume. By the time it executes, review_decision has been
    filled in from outside.

    That separation is what makes the pause survivable. The graph's state can be
    saved while it waits, and the wait can last a second or a week -- through a
    web request, a restart, an overnight queue -- because nothing is sitting
    blocked inside a running function.
    """
    decision = state.get("review_decision")

    if decision == "approved":
        # The reviewer may have edited the text. If they did, their version is
        # what the user sees; otherwise the original draft goes out.
        return {"final_answer": state.get("review_note") or state["draft_answer"]}

    if decision == "rejected":
        return {
            "final_answer": (
                "I wasn't able to answer that reliably from the annual reports. "
                "Please try rephrasing, or refer to the reports directly."
            )
        }

    # No decision recorded. This should not happen in normal operation, but a
    # node that assumes it cannot happen will crash the whole graph when it does.
    # Treating it as a rejection is the safe default: the user gets a polite
    # non-answer rather than an unreviewed one.
    return {
        "final_answer": (
            "This answer is awaiting review and can't be shown yet."
        )
    }


# ==============================================================================
# NODE 7 - FINALISE
# ==============================================================================

def finalise_node(state: GraphState) -> dict:
    """
    Copy the approved draft into final_answer for answers that needed no review.

    WHY A NODE FOR SOMETHING THIS SMALL
    -----------------------------------
    Because it gives every route through the graph exactly one place where
    final_answer is set on its way out.

    Without it, the grounded path would need generate_node to write final_answer
    directly -- and then draft_answer and final_answer would be the same thing on
    that path and different things on the review path. That kind of
    inconsistency is where confusing bugs live.

    Small nodes that make the graph regular are worth their cost.
    """
    return {"final_answer": state["draft_answer"]}