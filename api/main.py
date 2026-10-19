"""
================================================================================
WEB API - THE PIPELINE OVER HTTP
================================================================================

WHAT THIS FILE DOES
-------------------
Exposes the same graph run_query.py uses, as three HTTP endpoints:

    POST /ask                      ask a question
    POST /review/{thread_id}       approve or reject a paused answer
    GET  /health                   is the server up?

WHAT CHANGES COMPARED WITH run_query.py
---------------------------------------
Only one thing, and it is the thing this file exists to show.

In run_query.py the pause and the resume happen inside one run of the script: the
graph stops, the script asks the reviewer at the keyboard, the graph carries on.

Over HTTP they are two separate requests. /ask runs the graph until it pauses and
returns a thread_id. Minutes or days later, a reviewer calls /review with that
thread_id, and the graph picks up from where it stopped.

For that to work the paused state must outlive the first request -- and survive
the server restarting in between. So this file gives the graph a SqliteSaver,
which saves state to a file, instead of the in-memory MemorySaver. Nothing in the
graph itself changes.

HOW TO RUN
----------
    uvicorn api.main:app --reload

Then open http://127.0.0.1:8000/docs to try the endpoints in the browser.

================================================================================
"""

import sqlite3
import sys
import uuid
from pathlib import Path
from typing import List, Literal, Optional

from fastapi import FastAPI, HTTPException
from langgraph.checkpoint.sqlite import SqliteSaver
from pydantic import BaseModel

sys.path.append(str(Path(__file__).parent.parent))

import config
from graph.build_graph import build_graph
from graph.state import create_initial_state


# ==============================================================================
# SECTION 1 - THE GRAPH, BUILT ONCE
# ==============================================================================
# Building the graph is cheap, but the checkpointer holds a database connection,
# so we make one of each when the server starts and share them across requests.
#
# check_same_thread=False: the web server handles requests on several threads,
# and SQLite refuses by default to share a connection between them. SqliteSaver
# takes its own lock around every write, so sharing is safe here.
# ==============================================================================

config.CHECKPOINT_DB.parent.mkdir(parents=True, exist_ok=True)
checkpointer = SqliteSaver(sqlite3.connect(config.CHECKPOINT_DB, check_same_thread=False))
graph = build_graph(checkpointer=checkpointer)

app = FastAPI(
    title="Annual Report RAG",
    description="Year-over-year questions about annual reports, with human review.",
)


# ==============================================================================
# SECTION 2 - REQUEST AND RESPONSE SHAPES
# ==============================================================================
# FastAPI reads these pydantic classes to check incoming JSON and to write the
# documentation at /docs. A request missing "question" is rejected with a clear
# error before any of our code runs.
# ==============================================================================

class AskRequest(BaseModel):
    question: str
    company: str


class ReviewRequest(BaseModel):
    decision: Literal["approved", "rejected"]
    # Optional corrected answer. If given with "approved", it is shown instead of
    # the draft -- the same as editing the answer in run_query.py.
    note: Optional[str] = None


class RunResponse(BaseModel):
    """
    status
        "answered"   final_answer is ready to show.
        "blocked"    the input guardrail refused; final_answer explains why.
        "paused"     waiting for a reviewer. Call /review/{thread_id}.
    """
    thread_id: str
    status: Literal["answered", "blocked", "paused"]
    final_answer: Optional[str] = None
    draft_answer: Optional[str] = None
    grounding_verdict: Optional[str] = None
    unsupported_items: List[str] = []


def describe(thread_id: str, state: dict) -> RunResponse:
    """Turn the graph's state into the response the caller sees."""
    thread = {"configurable": {"thread_id": thread_id}}

    if not state.get("input_allowed", True):
        status = "blocked"
    elif graph.get_state(thread).next:
        status = "paused"
    else:
        status = "answered"

    return RunResponse(
        thread_id=thread_id,
        status=status,
        # While paused there is no final answer yet; only the reviewer sees the
        # draft. Showing it to the user would defeat the review.
        final_answer=None if status == "paused" else state.get("final_answer"),
        draft_answer=state.get("draft_answer") if status == "paused" else None,
        grounding_verdict=state.get("grounding_verdict"),
        unsupported_items=state.get("unsupported_items") or [],
    )


# ==============================================================================
# SECTION 3 - THE ENDPOINTS
# ==============================================================================
# These are plain "def", not "async def". The graph makes blocking calls (OpenAI,
# the vector store, the reranker), and FastAPI runs plain functions on a worker
# thread so one slow question does not freeze the whole server.
# ==============================================================================

@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/ask", response_model=RunResponse)
def ask(request: AskRequest) -> RunResponse:
    if request.company not in config.COMPANIES:
        raise HTTPException(
            status_code=400,
            detail=f"company must be one of: {', '.join(config.COMPANIES)}",
        )

    thread_id = str(uuid.uuid4())
    thread = {"configurable": {"thread_id": thread_id}}
    state = graph.invoke(
        create_initial_state(question=request.question, company=request.company),
        thread,
    )
    return describe(thread_id, state)


@app.post("/review/{thread_id}", response_model=RunResponse)
def review(thread_id: str, request: ReviewRequest) -> RunResponse:
    thread = {"configurable": {"thread_id": thread_id}}
    snapshot = graph.get_state(thread)

    # No saved state means the id is wrong. Saved state with nothing queued means
    # the run already finished -- reviewing it again would be meaningless.
    if not snapshot.values:
        raise HTTPException(status_code=404, detail="no run with that thread_id")
    if not snapshot.next:
        raise HTTPException(status_code=409, detail="that run is not waiting for review")

    # Same two steps as run_query.py: write the decision into the saved state,
    # then resume with None so the graph continues rather than starting over.
    graph.update_state(thread, {"review_decision": request.decision,
                                "review_note": request.note})
    state = graph.invoke(None, thread)
    return describe(thread_id, state)
