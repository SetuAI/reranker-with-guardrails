"""
================================================================================
GRAPH STEP 3 - BUILDING THE GRAPH
================================================================================

WHAT THIS FILE DOES
-------------------
Connects the nodes together and decides what runs when.

state.py described the shape of the data. nodes.py described what each step does.
This file is the only place that knows the ORDER.

THE SHAPE OF THE PIPELINE
-------------------------
                        input guardrail
                              |
                    +---------+---------+
                    |                   |
              blocked                 allowed
                    |                   |
                    |               retrieve
                    |                   |
                    |                rerank
                    |                   |
                    |               generate
                    |                   |
                    |            output guardrail
                    |                   |
                    |         +---------+---------+
                    |         |                   |
                    |     grounded            needs review
                    |         |                   |
                    |     finalise         [ GRAPH PAUSES ]
                    |         |                   |
                    |         |            human review
                    |         |                   |
                    +---------+---------+---------+
                                  |
                                 END

TWO KINDS OF EDGE
-----------------
A NORMAL edge always goes to the same next node. "After retrieving, always
rerank."

A CONDITIONAL edge looks at the state and chooses. "After the output guardrail,
go to finalise OR to human review, depending on the grounding verdict."

The two forks above -- blocked/allowed and grounded/needs-review -- are the only
conditional edges in this graph. Everything else is a straight line.

THE INTERRUPT
-------------
The single most important line in this file is interrupt_before=["human_review"].

It tells LangGraph to stop the run just before that node and hand control back to
whoever called it. The graph does not block, sleep, or wait in a loop -- it
returns, with its state saved.

Later, the caller can resume from exactly that point. The pause can last a
second, or a week, across a web request or a restart, because nothing is sitting
frozen inside a running function.

That is what makes human-in-the-loop practical rather than a demonstration trick.

WHY A CHECKPOINTER IS REQUIRED
------------------------------
Pausing is only useful if the state survives the pause. A checkpointer is what
saves it.

We use MemorySaver, which keeps state in memory. It is enough for this project
and it disappears when the program exits. A production system would use a
database-backed checkpointer so a paused run survives a restart -- the graph code
would not change at all, only this one object.

Without a checkpointer, LangGraph will not allow an interrupt, because a pause
with nothing to resume from is meaningless.

HOW TO RUN
----------
    python graph/build_graph.py

Prints the structure without running anything.

================================================================================
"""

import sys
from pathlib import Path

from langgraph.graph import END, START, StateGraph
from langgraph.checkpoint.memory import MemorySaver

sys.path.append(str(Path(__file__).parent.parent))

import config
from graph.state import GraphState
from graph.nodes import (
    input_guardrail_node,
    retrieve_node,
    rerank_node,
    generate_node,
    output_guardrail_node,
    human_review_node,
    finalise_node,
)
from guardrails.factual import GROUNDED


# ==============================================================================
# SECTION 1 - THE ROUTING DECISIONS
# ==============================================================================
# These functions are not nodes. They do no work and change nothing.
#
# Each one reads the state and returns a STRING -- the name of the route to take.
# LangGraph looks that string up in a map we provide below and sends the run to
# the matching node.
#
# Keeping routing separate from the nodes themselves is a deliberate split. A
# node that both did its job AND decided what came next would mean changing the
# routing rule required editing the working logic. As written, every routing rule
# in the whole pipeline is visible in these two short functions.
# ==============================================================================

def route_after_input_guardrail(state: GraphState) -> str:
    """
    Decide whether the question proceeds or stops here.

    Returns "allowed" or "blocked".

    Note this reads a field the guardrail node already wrote. The decision was
    made there; this function only reports which way to go. That is why it can be
    two lines long.
    """
    return "allowed" if state["input_allowed"] else "blocked"


def route_after_output_guardrail(state: GraphState) -> str:
    """
    Decide whether the answer goes straight out or waits for a person.

    Returns "grounded" or "needs_review".

    THIS IS THE MOST CONSEQUENTIAL LINE IN THE PROJECT
    --------------------------------------------------
    Everything about human-in-the-loop comes down to this single comparison.

    It is worth seeing how narrow it is. There is no clever logic deciding when a
    human is needed -- just one verdict, produced by one guardrail, checked once.

    The reason it can be this simple is that all the judgement already happened in
    guardrails/factual.py. Routing should be the last, smallest step, not the
    place where decisions get made.

    config.INTERRUPT_ON_LOW_GROUNDING lets us turn the pause off entirely, which
    is useful when running a batch of questions where nobody is available to
    review anything.
    """
    if not config.INTERRUPT_ON_LOW_GROUNDING:
        return "grounded"

    return "grounded" if state["grounding_verdict"] == GROUNDED else "needs_review"


# ==============================================================================
# SECTION 2 - BUILDING THE GRAPH
# ==============================================================================

def build_graph(enable_interrupt: bool = True):
    """
    Assemble the pipeline and return something runnable.

    PARAMETERS
    ----------
    enable_interrupt
        When True, the graph pauses before human review. When False it runs
        straight through -- useful for testing, and for batch runs where nobody
        is waiting to review anything.

    RETURNS
    -------
    A compiled graph, ready to be invoked.
    """
    # StateGraph is the builder. Passing GraphState tells it the shape of the
    # state that flows through, so it knows how to merge each node's returned
    # fields into it.
    builder = StateGraph(GraphState)

    # ---------------------------------------------------------------------
    # Register the nodes
    # ---------------------------------------------------------------------
    # The first argument is a NAME. That name is how edges refer to the node, and
    # how the interrupt identifies where to stop. It is a plain string, so a typo
    # produces an error at build time -- which is the good moment for it.
    builder.add_node("input_guardrail", input_guardrail_node)
    builder.add_node("retrieve", retrieve_node)
    builder.add_node("rerank", rerank_node)
    builder.add_node("generate", generate_node)
    builder.add_node("output_guardrail", output_guardrail_node)
    builder.add_node("human_review", human_review_node)
    builder.add_node("finalise", finalise_node)

    # ---------------------------------------------------------------------
    # Where the run begins
    # ---------------------------------------------------------------------
    # START is a special marker meaning "the entry point". Every graph needs
    # exactly one edge leading out of it.
    builder.add_edge(START, "input_guardrail")

    # ---------------------------------------------------------------------
    # FORK 1 - did the question pass the input guardrail?
    # ---------------------------------------------------------------------
    # add_conditional_edges takes three things:
    #
    #   1. the node we are leaving
    #   2. the function that decides which way to go
    #   3. a map from that function's return value to a destination node
    #
    # A blocked question goes straight to END. It never touches retrieval,
    # generation, or the output guardrail -- which is exactly why the input
    # guardrail is worth having at the front.
    builder.add_conditional_edges(
        "input_guardrail",
        route_after_input_guardrail,
        {
            "allowed": "retrieve",
            "blocked": END,
        },
    )

    # ---------------------------------------------------------------------
    # The straight run through the middle
    # ---------------------------------------------------------------------
    # No decisions here. Retrieval always leads to reranking, which always leads
    # to generation, which always leads to the output guardrail.
    builder.add_edge("retrieve", "rerank")
    builder.add_edge("rerank", "generate")
    builder.add_edge("generate", "output_guardrail")

    # ---------------------------------------------------------------------
    # FORK 2 - is the answer grounded, or does it need a person?
    # ---------------------------------------------------------------------
    # This is where human-in-the-loop enters the graph.
    builder.add_conditional_edges(
        "output_guardrail",
        route_after_output_guardrail,
        {
            "grounded": "finalise",
            "needs_review": "human_review",
        },
    )

    # ---------------------------------------------------------------------
    # Both routes converge at the end
    # ---------------------------------------------------------------------
    builder.add_edge("finalise", END)
    builder.add_edge("human_review", END)

    # ---------------------------------------------------------------------
    # Compile
    # ---------------------------------------------------------------------
    # Up to this point we have only described the graph. Compiling turns that
    # description into something runnable, and checks it makes sense -- that
    # every named node exists, and that there are no unreachable nodes.
    #
    # ABOUT interrupt_before
    # ----------------------
    # This list names the nodes to stop before. When the run reaches one, it
    # returns instead of continuing.
    #
    # Note it is interrupt_BEFORE, not after. We stop before human_review because
    # that node's job is to APPLY a reviewer's decision -- so it must not run
    # until a decision exists.
    #
    # ABOUT THE CHECKPOINTER
    # ----------------------
    # MemorySaver stores the paused state so the run can be picked up again.
    # LangGraph requires one for interrupts, because a pause with nothing to
    # resume from would be pointless.
    #
    # It also means every run needs an ID, so the graph knows which saved state
    # to resume. That is the thread_id passed in when the graph is invoked.
    return builder.compile(
        checkpointer=MemorySaver(),
        interrupt_before=["human_review"] if enable_interrupt else [],
    )


# ==============================================================================
# SECTION 3 - RUNNING THIS FILE DIRECTLY
# ==============================================================================

if __name__ == "__main__":
    print("=" * 74)
    print("BUILDING THE GRAPH")
    print("=" * 74)
    print()

    graph = build_graph()

    print("  Compiled successfully.")
    print()
    print("  nodes:")
    for name in graph.get_graph().nodes:
        print(f"    {name}")

    print()
    print("  interrupt before: human_review")
    print(f"  interrupt enabled in config: {config.INTERRUPT_ON_LOW_GROUNDING}")
    print()

    # LangGraph can draw the structure as a diagram in Mermaid format. Useful for
    # confirming the wiring matches what you intended -- a graph that compiles is
    # not necessarily a graph that is connected the way you think.
    print("-" * 74)
    print("STRUCTURE")
    print("-" * 74)
    print()

    try:
        print(graph.get_graph().draw_mermaid())
    except Exception as error:
        print(f"  (could not draw the diagram: {type(error).__name__})")

    print()