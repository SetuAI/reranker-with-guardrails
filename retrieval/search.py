"""
================================================================================
RETRIEVAL STEP 1 - VECTOR SEARCH
================================================================================

WHAT THIS FILE DOES
-------------------
Given a question, finds the chunks in the database whose meaning is closest to
it.

This is the "retrieval" in retrieval augmented generation. Nothing here talks to
a language model beyond turning the question into numbers -- no answer is written
at this stage. We are only finding relevant text.

HOW SEARCHING BY MEANING WORKS
------------------------------
During ingestion, every chunk was converted into an embedding: a list of numbers
representing its meaning, positioned so that similar meanings sit close together.

To search, we convert the QUESTION into an embedding using the same model, then
ask the database which chunk embeddings sit nearest to it.

The important consequence is that matching does not depend on shared words. A
question asking "how exposed is the company to exchange rates?" will find a chunk
saying "a significant portion of revenue is denominated in foreign currency",
even though the two share almost no vocabulary.

THE PROBLEM THIS FILE EXISTS TO SOLVE
-------------------------------------
Search by meaning has a weakness, and in this project it is severe.

Our database holds two annual reports per company, one year apart. Those two
reports are written in nearly identical language: the same section headings, the
same sentence structures, the same boilerplate about risks and strategy. Only the
numbers differ, and often not by much.

So when a question is asked, chunks from both years come back mixed together,
ranked essentially at random relative to each other -- because in terms of
MEANING they genuinely are almost the same text. There is no threshold to tune
and no better embedding model that fixes this. The information distinguishing
them is not in the meaning at all.

Ask "what was the operating margin?" and the model may confidently answer using
last year's figure, with nothing in the output to suggest anything went wrong.

THE FIX
-------
We do not ask search to work out the year. We attached the year to every chunk
during ingestion, and here we FILTER by it -- searching within FY2025 only, then
within FY2026 only, and comparing the two sets of results.

Filtering is exact and cannot be wrong. Similarity is approximate and, on this
corpus, unreliable. Use each for what it is good at: filtering to decide WHICH
DOCUMENTS are eligible, similarity to decide WHICH PARTS of them are relevant.

That division of labour is the single most useful idea in this file.

HOW TO RUN
----------
    python retrieval/search.py

Running it directly demonstrates the problem and the fix side by side. Ingestion
must have been run first.

================================================================================
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional

from openai import OpenAI

sys.path.append(str(Path(__file__).parent.parent))

import config
from ingestion.build_index import get_collection


client = OpenAI(api_key=config.OPENAI_API_KEY)


# ==============================================================================
# SECTION 1 - TURNING THE QUESTION INTO NUMBERS
# ==============================================================================

def embed_question(question: str) -> List[float]:
    """
    Convert a question into an embedding so it can be compared with chunks.

    THE MODEL MUST BE THE SAME ONE USED DURING INGESTION
    ----------------------------------------------------
    This uses config.EMBEDDING_MODEL, exactly as ingestion did, and that is not a
    coincidence to be casually changed.

    Different embedding models place text in completely different number spaces.
    Comparing an embedding from one model against embeddings from another is
    meaningless -- like comparing a temperature in Celsius against one in
    Fahrenheit without converting.

    Nothing would crash if you changed it. Search would simply return unrelated
    chunks forever, with no error to explain why. If you ever change the
    embedding model, you must rebuild the whole index.
    """
    response = client.embeddings.create(
        model=config.EMBEDDING_MODEL,
        input=[question],
    )

    # We sent one text, so we take the first (and only) result.
    return response.data[0].embedding


# ==============================================================================
# SECTION 2 - BUILDING THE FILTER
# ==============================================================================

def build_filter(company: Optional[str] = None,
                 fiscal_year: Optional[str] = None) -> Optional[Dict]:
    """
    Build the metadata filter that restricts which chunks are eligible.

    PARAMETERS
    ----------
    company      restrict to one company, or None for all
    fiscal_year  restrict to one year, or None for all

    RETURNS
    -------
    A filter dictionary in the form Chroma expects, or None when no restriction
    is wanted.

    ABOUT Optional[str]
    -------------------
    Optional[str] means "either a string, or None". The "= None" makes the
    argument optional to pass. So all four of these are valid:

        build_filter()
        build_filter(company="Infosys")
        build_filter(fiscal_year="FY2026")
        build_filter(company="Infosys", fiscal_year="FY2026")

    ABOUT THE FILTER SYNTAX
    -----------------------
    Chroma expects conditions written as nested dictionaries:

        {"company": {"$eq": "Infosys"}}          company equals Infosys

    When there is more than one condition they must be wrapped in $and:

        {"$and": [ {...}, {...} ]}               both conditions must hold

    And -- this is the part that catches people -- $and requires at least TWO
    conditions. Passing it a list containing one condition is an error, not a
    harmless case. That is why the code below handles the one-condition case
    separately rather than always wrapping.

    HOW THIS DIFFERS FROM SIMILARITY
    --------------------------------
    A filter is a hard rule. A chunk either matches or it does not, and a chunk
    that fails the filter cannot appear in the results no matter how similar it
    looks.

    Similarity is a soft ranking. Everything that survives the filter gets scored
    and ordered.

    Filtering first, then ranking, is what makes year-by-year comparison possible
    at all.
    """
    conditions = []

    if company is not None:
        conditions.append({"company": {"$eq": company}})

    if fiscal_year is not None:
        conditions.append({"fiscal_year": {"$eq": fiscal_year}})

    # No restrictions -- search the whole database.
    if not conditions:
        return None

    # Exactly one condition -- return it on its own, without $and.
    if len(conditions) == 1:
        return conditions[0]

    # Two or more -- combine them, requiring all to hold.
    return {"$and": conditions}


# ==============================================================================
# SECTION 3 - THE SEARCH
# ==============================================================================

def search(question: str,
           company: Optional[str] = None,
           fiscal_year: Optional[str] = None,
           top_k: Optional[int] = None) -> List[Dict]:
    """
    Find the chunks most relevant to a question.

    PARAMETERS
    ----------
    question     what the user asked
    company      restrict to one company, or None for all
    fiscal_year  restrict to one year, or None for all
    top_k        how many chunks to return. Defaults to the value in config.

    RETURNS
    -------
    A list of result dictionaries, most relevant first. Each holds the chunk
    text, its metadata, and a similarity score.

    ABOUT THE DEFAULT FOR top_k
    ---------------------------
    Notice the default is None rather than config.VECTOR_SEARCH_TOP_K written
    directly in the function signature.

    The reason is a genuine Python trap: default argument values are evaluated
    ONCE, when the function is first defined, not each time it is called. If
    config.VECTOR_SEARCH_TOP_K were used as the default directly, the value would
    be frozen at import time, and changing the config later in the same session
    would have no effect -- a very confusing thing to debug.

    Using None and resolving it inside the function reads the config fresh on
    every call.

    WHY WE ASK FOR 20 RATHER THAN 5
    -------------------------------
    config.VECTOR_SEARCH_TOP_K is deliberately generous. Vector search is fast
    but imprecise: the genuinely best chunk is often sitting at position 11
    rather than position 1.

    We are not trying to get the right answer here. We are trying to make sure
    the right answer is somewhere in the pile, so the reranker in the next file
    can find it. Casting a wide net now is what makes careful ranking possible
    later.
    """
    if top_k is None:
        top_k = config.VECTOR_SEARCH_TOP_K

    collection = get_collection()
    question_embedding = embed_question(question)
    where_filter = build_filter(company, fiscal_year)

    # Ask Chroma for the nearest chunks.
    #
    # query_embeddings takes a LIST of embeddings, because Chroma supports
    # searching for several questions at once. We have one, so we wrap it in a
    # list -- and the results come back wrapped the same way, which is why every
    # result field is indexed with [0] below.
    #
    # include says which parts of each record we want back. Chroma returns only
    # the ids by default, so without this we would get no text.
    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=top_k,
        where=where_filter,
        include=["documents", "metadatas", "distances"],
    )

    # Reshape Chroma's output into a plain list of dictionaries.
    #
    # Chroma returns parallel lists: one list of texts, one of metadata, one of
    # distances, all in matching order. That shape is awkward to work with, so we
    # zip them back together into one dictionary per result.
    formatted = []

    for text, metadata, distance in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        formatted.append({
            "text": text,
            "company": metadata["company"],
            "fiscal_year": metadata["fiscal_year"],
            "year_label": metadata["year_label"],
            "position": metadata["position"],

            # CONVERTING DISTANCE INTO SIMILARITY
            # -----------------------------------
            # Chroma reports DISTANCE, where smaller means more alike. We prefer
            # SIMILARITY, where larger means more alike, because it reads more
            # naturally and matches how the reranker scores things later.
            #
            # With cosine distance the two are related by: similarity = 1 -
            # distance. Since we set the collection to use cosine during
            # ingestion, this conversion is correct here.
            "score": 1 - distance,
        })

    return formatted


# ==============================================================================
# SECTION 4 - SEARCHING BOTH YEARS
# ==============================================================================

def search_both_years(question: str,
                      company: str,
                      top_k: Optional[int] = None) -> Dict[str, List[Dict]]:
    """
    Run the same question separately against each financial year.

    RETURNS
    -------
    A dictionary keyed by fiscal year, each holding that year's results:

        {"FY2025": [...], "FY2026": [...]}

    THIS FUNCTION IS THE HEART OF THE PROJECT
    -----------------------------------------
    Everything about comparing years rests on it.

    The naive approach would be one search across both years, hoping the results
    happen to include some of each. On this corpus that fails, and it fails
    quietly: the two years read almost identically, so the results skew
    arbitrarily towards whichever year's phrasing happened to sit marginally
    closer, and you get five chunks from FY2026 and none from FY2025 with nothing
    to indicate the imbalance.

    Running two separate filtered searches guarantees balanced evidence. We get
    the best FY2025 chunks AND the best FY2026 chunks, and only then ask what
    changed between them.

    The cost is one extra embedding call per question, which is negligible.
    """
    return {
        fiscal_year: search(question, company=company,
                            fiscal_year=fiscal_year, top_k=top_k)
        for fiscal_year in config.FISCAL_YEARS
    }


# ==============================================================================
# SECTION 5 - RUNNING THIS FILE DIRECTLY
# ==============================================================================
# The demonstration below shows the problem first and the fix second, using the
# same question for both so the difference is attributable to the filter alone.
# ==============================================================================

def show_results(results: List[Dict], limit: int = 8) -> None:
    """Print search results as a compact table."""
    for rank, result in enumerate(results[:limit], start=1):
        # Collapse any line breaks so each chunk stays on one line.
        preview = result["text"].replace("\n", " ")[:58]
        print(f"    {rank:>2}. [{result['fiscal_year']}] "
              f"{result['score']:.3f}  {preview}...")


if __name__ == "__main__":
    QUESTION = "What are the main risks the company faces from currency movements?"
    COMPANY = "Infosys"

    print("=" * 74)
    print("VECTOR SEARCH")
    print("=" * 74)
    print()
    print(f"  question   {QUESTION}")
    print(f"  company    {COMPANY}")

    # ---------------------------------------------------------------------
    # WITHOUT the year filter
    # ---------------------------------------------------------------------
    print()
    print("-" * 74)
    print("SEARCHING BOTH YEARS AT ONCE (no year filter)")
    print("-" * 74)
    print()

    mixed = search(QUESTION, company=COMPANY)
    show_results(mixed)

    # Count how many of the results came from each year. This number is the
    # whole point of the demonstration.
    from collections import Counter
    year_counts = Counter(r["fiscal_year"] for r in mixed)

    print()
    print(f"    of {len(mixed)} results: "
          + ", ".join(f"{year} = {count}" for year, count in sorted(year_counts.items())))
    print()
    print("    The two years are interleaved, and the split is arbitrary. The")
    print("    scores sit very close together because the underlying text really")
    print("    is nearly the same in both reports. There is no threshold that")
    print("    separates them, because the difference is not one of meaning.")

    # ---------------------------------------------------------------------
    # WITH the year filter
    # ---------------------------------------------------------------------
    print()
    print("-" * 74)
    print("SEARCHING EACH YEAR SEPARATELY (year filter applied)")
    print("-" * 74)

    by_year = search_both_years(QUESTION, company=COMPANY)

    for fiscal_year, results in by_year.items():
        print()
        print(f"  {fiscal_year}")
        show_results(results, limit=5)

    print()
    print("    Now each year is represented on its own terms, and we have")
    print("    balanced evidence from both. Only now is it possible to ask")
    print("    what actually changed between them.")
    print()