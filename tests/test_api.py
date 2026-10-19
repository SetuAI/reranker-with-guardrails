"""
The web API, end to end, with the working nodes stubbed out.

The important test is the last one: a run paused by one request is resumed by a
later request, after the graph has been rebuilt from scratch -- which is what a
server restart looks like. That only works because the state is in SQLite.
"""

import importlib
import sqlite3
import sys
import uuid

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("langgraph.checkpoint.sqlite")

from fastapi.testclient import TestClient  # noqa: E402
from langgraph.checkpoint.sqlite import SqliteSaver  # noqa: E402

import test_graph  # noqa: E402  -- registers the retrieval stand-ins
from graph import build_graph as graph_module  # noqa: E402
from graph.state import create_initial_state  # noqa: E402
from guardrails.factual import GROUNDED, NEEDS_REVIEW  # noqa: E402


@pytest.fixture
def api(monkeypatch, tmp_path):
    """Load api.main against a temporary database, with stubbed nodes."""
    def load(verdict=GROUNDED):
        test_graph.stub_pipeline(monkeypatch, verdict=verdict)
        monkeypatch.setattr(graph_module.config, "CHECKPOINT_DB", tmp_path / "cp.sqlite")
        sys.modules.pop("api.main", None)
        module = importlib.import_module("api.main")
        return module, TestClient(module.app)
    return load


def test_unknown_company_is_rejected(api):
    _, client = api()
    response = client.post("/ask", json={"question": "q", "company": "Acme"})
    assert response.status_code == 400


def test_grounded_answer_comes_straight_back(api):
    _, client = api(verdict=GROUNDED)
    body = client.post("/ask", json={"question": "q", "company": "Infosys"}).json()
    assert body["status"] == "answered"
    assert body["final_answer"] == "draft"


def test_paused_answer_hides_the_draft_from_the_user(api):
    _, client = api(verdict=NEEDS_REVIEW)
    body = client.post("/ask", json={"question": "q", "company": "Infosys"}).json()
    assert body["status"] == "paused"
    assert body["final_answer"] is None


def test_review_resumes_the_paused_run(api):
    _, client = api(verdict=NEEDS_REVIEW)
    thread_id = client.post("/ask", json={"question": "q", "company": "Infosys"}).json()["thread_id"]

    body = client.post(f"/review/{thread_id}",
                       json={"decision": "approved", "note": "corrected"}).json()
    assert body["status"] == "answered"
    assert body["final_answer"] == "corrected"

    # A second review of the same run is refused.
    again = client.post(f"/review/{thread_id}", json={"decision": "approved"})
    assert again.status_code == 409


def test_review_of_unknown_run_is_404(api):
    _, client = api()
    assert client.post("/review/nope", json={"decision": "approved"}).status_code == 404


def test_paused_run_survives_a_restart(monkeypatch, tmp_path):
    test_graph.stub_pipeline(monkeypatch, verdict=NEEDS_REVIEW)
    db = tmp_path / "cp.sqlite"
    thread = {"configurable": {"thread_id": str(uuid.uuid4())}}

    # First "server": run until the pause, then throw everything away.
    first = graph_module.build_graph(checkpointer=SqliteSaver(sqlite3.connect(db, check_same_thread=False)))
    first.invoke(create_initial_state(question="q", company="Infosys"), thread)
    del first

    # Second "server": a brand-new graph and connection, same file.
    second = graph_module.build_graph(checkpointer=SqliteSaver(sqlite3.connect(db, check_same_thread=False)))
    assert second.get_state(thread).next == ("human_review",)

    second.update_state(thread, {"review_decision": "rejected"})
    state = second.invoke(None, thread)
    assert "wasn't able to answer" in state["final_answer"]
