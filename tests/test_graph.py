"""
The human-in-the-loop pause, tested end to end through the real graph.

Retrieval, generation and the guardrails are replaced with stubs so this runs
offline and in under a second. Everything that matters here -- the routing, the
interrupt, the checkpointer, resuming with a reviewer's decision -- is real.
"""

import sys
import types
import uuid

import pytest

pytest.importorskip("langgraph")

# graph.nodes imports the retrieval modules, which load the vector store and the
# cross-encoder. Neither is needed to test the graph's wiring, so stand-ins are
# registered before the import.
for name, functions in {
    "retrieval.search": {"search_both_years": lambda **kwargs: {}},
    "retrieval.rerank": {"rerank": lambda question, candidates: []},
}.items():
    if name not in sys.modules:
        sys.modules[name] = types.SimpleNamespace(**functions)

from graph import build_graph as graph_module  # noqa: E402
from graph.state import create_initial_state  # noqa: E402
from guardrails.factual import GROUNDED, NEEDS_REVIEW  # noqa: E402


def stub_pipeline(monkeypatch, input_allowed=True, verdict=GROUNDED):
    """Swap every working node for a stub, leaving the graph's wiring intact."""
    def input_guardrail(state):
        if input_allowed:
            return {"input_allowed": True}
        return {"input_allowed": False, "input_block_reason": "stub",
                "final_answer": "refused"}

    def output_guardrail(state):
        return {"grounding_verdict": verdict, "grounding_score": 0.0,
                "unsupported_items": [], "review_required": verdict != GROUNDED}

    monkeypatch.setattr(graph_module, "input_guardrail_node", input_guardrail)
    monkeypatch.setattr(graph_module, "retrieve_node",
                        lambda state: {"candidate_chunks": {}})
    monkeypatch.setattr(graph_module, "rerank_node",
                        lambda state: {"reranked_chunks": {}})
    monkeypatch.setattr(graph_module, "generate_node",
                        lambda state: {"draft_answer": "draft"})
    monkeypatch.setattr(graph_module, "output_guardrail_node", output_guardrail)
    monkeypatch.setattr(graph_module.config, "INTERRUPT_ON_LOW_GROUNDING", True)


def start(graph):
    thread = {"configurable": {"thread_id": str(uuid.uuid4())}}
    state = graph.invoke(create_initial_state(question="q", company="Infosys"), thread)
    return state, thread


def test_blocked_question_never_reaches_retrieval(monkeypatch):
    stub_pipeline(monkeypatch, input_allowed=False)
    monkeypatch.setattr(graph_module, "retrieve_node",
                        lambda state: pytest.fail("retrieval ran"))
    state, _ = start(graph_module.build_graph())
    assert state["final_answer"] == "refused"


def test_grounded_answer_finishes_without_pausing(monkeypatch):
    stub_pipeline(monkeypatch, verdict=GROUNDED)
    graph = graph_module.build_graph()
    _, thread = start(graph)
    assert not graph.get_state(thread).next


def test_unverified_answer_pauses_and_resumes_with_the_decision(monkeypatch):
    stub_pipeline(monkeypatch, verdict=NEEDS_REVIEW)
    graph = graph_module.build_graph()
    _, thread = start(graph)

    assert graph.get_state(thread).next == ("human_review",)

    graph.update_state(thread, {"review_decision": "approved", "review_note": None})
    state = graph.invoke(None, thread)

    assert state["review_decision"] == "approved"
    assert not graph.get_state(thread).next
