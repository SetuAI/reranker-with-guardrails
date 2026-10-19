"""
================================================================================
EVALUATION - DOES THE PIPELINE DO WHAT WE EXPECT?
================================================================================

WHAT THIS FILE DOES
-------------------
Runs every question in evaluation/questions.json through the real pipeline and
compares what happened with what was expected. Each question is labelled with one
of three outcomes:

    blocked    an input guardrail should refuse it
    answered   it should be answered without a person
    review     it should pause for a human reviewer

WHY THIS EXISTS
---------------
GROUNDING_THRESHOLD in config.py decides how many answers go to review. Without a
set of questions with known expected outcomes, the only way to pick it is to try
a few questions by hand and guess.

With this file you can try a threshold and see, in numbers, what it does:

    python run_evaluation.py --threshold 0.90
    python run_evaluation.py --threshold 0.95

A good threshold sends the "review" questions to review WITHOUT also sending the
"answered" ones. Too low and invented answers get through; too high and every
answer lands in the review queue, where nobody reads them properly.

THE LABELS ARE A STARTING POINT
-------------------------------
The questions supplied are examples. The "answered" and "review" labels are
best guesses about what these reports contain. Check them against the reports,
change any that are wrong, and add your own -- the evaluation is only as good as
its labels.

COST
----
Every question that is not blocked makes real API calls (embedding, generation,
grounding). Ten questions cost a few cents.

HOW TO RUN
----------
    python run_evaluation.py
    python run_evaluation.py --threshold 0.95

Requires the index: run run_ingestion.py first.

================================================================================
"""

import argparse
import json
import sys
import uuid
from collections import Counter
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

import config
from graph.build_graph import build_graph
from graph.state import create_initial_state


QUESTIONS_FILE = Path(__file__).parent / "evaluation" / "questions.json"


def outcome_of(graph, question: str, company: str) -> tuple:
    """
    Run one question and report what happened: blocked, answered or review.

    The graph pauses before review, exactly as it does in run_query.py. We never
    resume it -- a paused run IS the outcome we are measuring.
    """
    thread = {"configurable": {"thread_id": str(uuid.uuid4())}}
    state = graph.invoke(create_initial_state(question=question, company=company), thread)

    if not state["input_allowed"]:
        return "blocked", state.get("input_block_reason")
    if graph.get_state(thread).next:
        return "review", "; ".join(state.get("unsupported_items") or [])
    return "answered", ""


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the labelled evaluation questions.")
    parser.add_argument("--threshold", type=float, default=None,
                        help="override GROUNDING_THRESHOLD for this run only")
    args = parser.parse_args()

    # Overriding the value on the config module works because the grounding
    # check reads config.GROUNDING_THRESHOLD each time it runs.
    if args.threshold is not None:
        config.GROUNDING_THRESHOLD = args.threshold

    questions = json.loads(QUESTIONS_FILE.read_text())
    graph = build_graph()

    print("=" * 78)
    print(f"EVALUATION    {len(questions)} questions    "
          f"GROUNDING_THRESHOLD = {config.GROUNDING_THRESHOLD}")
    print("=" * 78)
    print()
    print(f"  {'':2} {'expected':9} {'got':9} question")
    print("  " + "-" * 74)

    results = []
    for item in questions:
        got, detail = outcome_of(graph, item["question"], item["company"])
        correct = got == item["expected"]
        results.append((item["expected"], got))

        mark = "ok" if correct else "XX"
        print(f"  {mark:2} {item['expected']:9} {got:9} {item['question'][:52]}")
        if not correct and detail:
            print(f"  {'':22} {detail[:52]}")

    # A table of expected vs actual. The diagonal is what went right; everything
    # off it tells you which way the threshold is wrong.
    counts = Counter(results)
    outcomes = ["blocked", "answered", "review"]

    print()
    print("  expected \\ got   " + "".join(f"{o:>10}" for o in outcomes))
    for expected in outcomes:
        row = "".join(f"{counts[(expected, got)]:>10}" for got in outcomes)
        print(f"  {expected:16} {row}")

    correct_total = sum(counts[(o, o)] for o in outcomes)
    print()
    print(f"  {correct_total} of {len(results)} as expected")
    print()
    print("  'answered' questions sent to review  ->  threshold may be too strict")
    print("  'review' questions answered          ->  threshold may be too lenient")
    print()


if __name__ == "__main__":
    main()
