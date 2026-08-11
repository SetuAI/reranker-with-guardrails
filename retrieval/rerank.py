"""
================================================================================
RETRIEVAL STEP 2 - RERANKING
================================================================================

WHAT THIS FILE DOES
-------------------
Takes the 20 candidate chunks that vector search returned and reduces them to the
best 5, using a slower but much more careful model.

WHY A SECOND RANKING STEP EXISTS AT ALL
---------------------------------------
It seems wasteful. Search already ranked the results -- why rank them again?

Because the two steps rank in fundamentally different ways, and the fast one is
not very good at it.

THE TWO KINDS OF MODEL
----------------------
Vector search uses what is called a BI-ENCODER. It converts the question into
numbers, converts each chunk into numbers SEPARATELY, and then compares the two
sets.

The key word is separately. Each chunk was turned into numbers during ingestion,
long before anyone asked a question. So the chunk's representation cannot depend
on the question -- it is a general-purpose summary of the chunk's meaning,
squeezed into a fixed list of numbers.

That is what makes search fast. Chunks are embedded once, and a search compares
against numbers already sitting in the database. Millions of chunks can be
searched in milliseconds.

It is also what makes search imprecise. A great deal of detail is lost when a
whole paragraph is compressed into one list of numbers, and nothing in that
compression knows which details the question cares about.

A CROSS-ENCODER works differently. It reads the question and one chunk TOGETHER,
as a single input, and produces a score for that specific pairing.

Because it sees both at once, it can weigh the parts of the chunk that matter for
this question. It is far more accurate.

It is also far slower, because nothing can be prepared in advance -- every
question-chunk pair needs its own pass through the model. Running it across
thousands of chunks would take minutes per question.

An analogy: the bi-encoder is like describing two people to a third party and
asking whether they would get along. The cross-encoder actually introduces them
and watches.

SO WE USE BOTH
--------------
    vector search    5,154 chunks  ->  20 candidates   fast, rough
    reranking           20 chunks  ->   5 final        slow, careful

Search casts a wide net cheaply. Reranking reads that small pile properly.

Neither works well alone. Search alone is imprecise. Reranking alone is far too
slow to run over the whole database.

WHY THE CUT FROM 20 TO 5 MATTERS AS MUCH AS THE REORDERING
----------------------------------------------------------
It is tempting to think of reranking as just putting things in a better order.
The discarding is at least as valuable.

Everything we pass to the model becomes context it must reason over. Fifteen
loosely relevant chunks do not sit there harmlessly -- they cost money, they
crowd the genuinely relevant text, and they give the model plausible-looking
material to draw a wrong conclusion from.

Being able to say "these 15 were not actually relevant" is the reranker's real
contribution.

A NOTE ON THIS MODEL
--------------------
The reranker runs on your own machine, not over the internet. The first time it
is used, roughly 90 MB is downloaded and cached; afterwards it works offline.

HOW TO RUN
----------
    python retrieval/rerank.py

Running it directly shows the same results before and after reranking, so the
change is visible. Ingestion must have been run first.

================================================================================
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional

from sentence_transformers import CrossEncoder

sys.path.append(str(Path(__file__).parent.parent))

import config
from retrieval.search import search, search_both_years


# ==============================================================================
# SECTION 1 - LOADING THE MODEL ONCE
# ==============================================================================
# Loading a model means reading its weights from disk into memory, which takes a
# few seconds. We do it once and reuse it for every question.
#
# The pattern below is called lazy loading: the model is not loaded when this
# file is imported, but the first time it is actually needed.
#
# That matters because other files import this one. Without lazy loading, merely
# importing rerank.py anywhere would pause for several seconds to load a model
# that might never be used -- for instance when a question is blocked by a
# guardrail and never reaches retrieval at all.
#
# The underscore at the start of _model is a Python convention meaning "this is
# internal to the file, do not use it from outside". Nothing enforces it; it is a
# signal to other readers.
# ==============================================================================

_model: Optional[CrossEncoder] = None


def get_model() -> CrossEncoder:
    """
    Return the reranking model, loading it on first use.

    The "global" keyword tells Python that we intend to change the _model
    variable defined outside this function. Without it, the assignment would
    create a new local variable that disappears when the function ends, and the
    model would be reloaded on every single call.
    """
    global _model

    if _model is None:
        print(f"    loading reranker ({config.RERANK_MODEL}) ...", flush=True)
        _model = CrossEncoder(config.RERANK_MODEL)

    return _model


# ==============================================================================
# SECTION 2 - RERANKING
# ==============================================================================

def rerank(question: str,
           results: List[Dict],
           top_k: Optional[int] = None) -> List[Dict]:
    """
    Score every result against the question and keep the best ones.

    PARAMETERS
    ----------
    question  what the user asked
    results   the candidates from vector search
    top_k     how many to keep. Defaults to config.RERANK_TOP_K.

    RETURNS
    -------
    The best results, most relevant first, each with an added rerank_score.

    WHY top_k DEFAULTS TO None
    --------------------------
    Same Python trap as in search.py: default argument values are evaluated once
    when the function is defined, not on each call. Writing
    config.RERANK_TOP_K directly as the default would freeze the value at import
    time, so changing the config later in the same session would silently have no
    effect.
    """
    if top_k is None:
        top_k = config.RERANK_TOP_K

    # Nothing to do if there are no candidates. Without this check the model
    # would be handed an empty list, which is a needless error.
    if not results:
        return []

    model = get_model()

    # Build the pairs to score.
    #
    # The cross-encoder expects a list of [question, text] pairs. Note that the
    # question is repeated in every pair -- that is the whole point. Each pair is
    # scored as its own complete input, which is why the model can weigh the
    # chunk against this specific question rather than in the abstract.
    pairs = [[question, result["text"]] for result in results]

    # Score every pair. This is the slow part -- one model pass per pair.
    #
    # The model handles the list in batches internally, so this is a single call
    # rather than a loop, but the work is still proportional to the number of
    # pairs. That is exactly why we hand it 20 and not 5,154.
    scores = model.predict(pairs)

    # Attach each score to its result.
    #
    # zip walks two lists together, giving one item from each on every turn. The
    # ordering guarantee matters here just as it did during embedding: score
    # number 7 belongs to result number 7.
    #
    # float() converts from the numpy number type the model returns into an
    # ordinary Python number, which prints more cleanly and avoids surprises if
    # the results are later saved to JSON.
    for result, score in zip(results, scores):
        result["rerank_score"] = float(score)

    # Sort by the new score, highest first.
    #
    #   key    tells sorted() what value to sort on for each item
    #   lambda a small unnamed function -- here, "given a result, return its
    #          rerank_score"
    #   reverse=True  sorts largest to smallest instead of smallest to largest
    reranked = sorted(results, key=lambda r: r["rerank_score"], reverse=True)

    # Keep only the top few. This slice is where the discarding happens, and it
    # is as important as the reordering above it.
    return reranked[:top_k]


# ==============================================================================
# SECTION 3 - SEARCH AND RERANK TOGETHER
# ==============================================================================

def search_and_rerank(question: str,
                      company: Optional[str] = None,
                      fiscal_year: Optional[str] = None) -> List[Dict]:
    """
    Run the full retrieval pipeline: search widely, then rank carefully.

    This is the function the rest of the project calls. It hides the two-stage
    detail behind one sensible call.
    """
    candidates = search(question, company=company, fiscal_year=fiscal_year)
    return rerank(question, candidates)


def search_and_rerank_both_years(question: str,
                                 company: str) -> Dict[str, List[Dict]]:
    """
    Run the full retrieval pipeline separately for each financial year.

    RETURNS
    -------
        {"FY2025": [...best 5...], "FY2026": [...best 5...]}

    WHY EACH YEAR IS RERANKED ON ITS OWN
    ------------------------------------
    We could combine both years' candidates into one pile and rerank them
    together. We deliberately do not.

    Reranking the combined pile would let one year dominate the final five. On
    this corpus that is a real risk, because the two reports describe the same
    topics in nearly identical language -- so the top five could easily come
    almost entirely from whichever year phrased things marginally closer to the
    question.

    The result would be a comparison built on evidence from one side. And it
    would look completely normal in the output, with nothing to indicate the
    imbalance.

    Reranking each year separately guarantees five good chunks from each. The
    comparison then has something real to compare.
    """
    by_year = search_both_years(question, company=company)

    return {
        fiscal_year: rerank(question, results)
        for fiscal_year, results in by_year.items()
    }


# ==============================================================================
# SECTION 4 - RUNNING THIS FILE DIRECTLY
# ==============================================================================

def show_comparison(question: str, candidates: List[Dict], reranked: List[Dict]) -> None:
    """
    Print the before and after side by side.

    The interesting thing to look for is MOVEMENT: which chunks climbed, which
    fell out of the top five entirely, and how far the reranker's opinion differs
    from the vector search's.
    """
    print()
    print("  BEFORE - as vector search ranked them")
    print("  " + "-" * 68)

    for rank, result in enumerate(candidates[:8], start=1):
        preview = result["text"].replace("\n", " ")[:48]
        print(f"    {rank:>2}. [{result['fiscal_year']}] "
              f"sim {result['score']:.3f}  {preview}...")

    print()
    print("  AFTER - as the reranker ranked them")
    print("  " + "-" * 68)

    # Work out where each surviving chunk USED to sit, so the movement is
    # visible. We match on the text itself, since that is unique per chunk.
    original_positions = {r["text"]: i + 1 for i, r in enumerate(candidates)}

    for rank, result in enumerate(reranked, start=1):
        preview = result["text"].replace("\n", " ")[:48]
        was = original_positions.get(result["text"], "?")
        print(f"    {rank:>2}. [{result['fiscal_year']}] "
              f"score {result['rerank_score']:>7.3f}  (was #{was})  {preview}...")

    print()
    print(f"    {len(candidates)} candidates in, {len(reranked)} kept, "
          f"{len(candidates) - len(reranked)} discarded.")


if __name__ == "__main__":
    QUESTION = "What are the main risks the company faces from currency movements?"
    COMPANY = "Infosys"

    print("=" * 74)
    print("RERANKING")
    print("=" * 74)
    print()
    print(f"  question   {QUESTION}")
    print(f"  company    {COMPANY}")
    print()

    # Retrieve candidates the ordinary way, then rerank them, so both lists are
    # available to compare.
    candidates = search(QUESTION, company=COMPANY)
    reranked = rerank(QUESTION, candidates)

    show_comparison(QUESTION, candidates, reranked)

    print()
    print("  Note the score scales are completely different. Similarity runs")
    print("  from 0 to 1. The reranker's score has no fixed range -- higher is")
    print("  better, and that is all it means. The two numbers cannot be")
    print("  compared with each other, only within their own column.")
    print()

    # ---------------------------------------------------------------------
    # The same thing, done per year -- which is how the project uses it.
    # ---------------------------------------------------------------------
    print("=" * 74)
    print("PER YEAR - HOW THE PROJECT ACTUALLY USES THIS")
    print("=" * 74)

    by_year = search_and_rerank_both_years(QUESTION, company=COMPANY)

    for fiscal_year, results in by_year.items():
        print()
        print(f"  {fiscal_year}")
        for rank, result in enumerate(results, start=1):
            preview = result["text"].replace("\n", " ")[:52]
            print(f"    {rank}. score {result['rerank_score']:>7.3f}  {preview}...")

    print()
    print("  Five strong chunks from each year, guaranteed. That balance is what")
    print("  makes a year-over-year comparison possible.")
    print()