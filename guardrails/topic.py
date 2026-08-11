"""
================================================================================
INPUT GUARDRAIL - TOPIC
================================================================================

WHAT THIS FILE DOES
-------------------
Decides whether an incoming question belongs to what this assistant is for.

Our assistant answers questions about two companies' annual reports. It should
handle questions about revenue, margins, risks, strategy, segments and how those
changed between financial years.

It should not answer questions about cooking, or write Python code, or recommend
whether to buy the shares. Those questions are not dangerous -- they are simply
not this system's job. Answering them costs money and gradually turns a
document research tool into a general chatbot nobody approved.

WHY THIS ONE IS NOT COPIED FROM THE EARLIER EXAMPLES
----------------------------------------------------
Earlier in this project we built a topic guardrail for a mutual fund customer
assistant. Its vocabulary was NAV, SIP, folio, redemption.

None of that is useful here. This is a research tool over annual reports, and the
vocabulary is revenue, margin, segment, risk factor.

That is worth stating plainly: A TOPIC GUARDRAIL DOES NOT TRANSFER BETWEEN
APPLICATIONS. The mechanism transfers -- keyword check, then similarity check --
but the content is specific to what you are building. Copying one from another
project and leaving its examples in place produces a guardrail that blocks your
real users and lets through things you never wanted.

THE APPROACH
------------
Two checks, in order:

    Check 1   Keyword matching       instant, free
    Check 2   Semantic similarity    ~50 ms, one embedding call

Check 1 looks for words we recognise. Finding one is enough to accept the
question and skip the API call entirely. Finding nothing proves nothing -- plenty
of legitimate questions avoid our vocabulary -- so in that case we move on to
Check 2, which compares meaning rather than spelling and makes the final call.

ONE BOUNDARY WORTH BEING DELIBERATE ABOUT
-----------------------------------------
"What did the company say about its margins?" is in scope. "Should I invest in
this company?" is not.

The difference is not topical -- both are about the same company and the same
financial subject. The difference is that the second asks for investment advice,
which this system is not qualified or permitted to give.

That line is drawn explicitly in the reference examples and the refusal message
below, rather than being left for the checks to infer.

HOW TO RUN
----------
    python guardrails/topic.py

================================================================================
"""

import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np
from openai import OpenAI

sys.path.append(str(Path(__file__).parent.parent))

import config


client = OpenAI(api_key=config.OPENAI_API_KEY)


# ==============================================================================
# SECTION 1 - DEFINING THE DOMAIN
# ==============================================================================
# We write the domain down twice, because the two checks consume it differently.
# ==============================================================================

# Used by Check 1: individual words that signal an annual report question.
DOMAIN_KEYWORDS = [
    "revenue", "margin", "profit", "loss", "ebitda", "earnings",
    "segment", "guidance", "outlook", "risk", "risks", "headcount",
    "attrition", "employees", "clients", "customers", "orders",
    "growth", "expenses", "capex", "cash flow", "dividend",
    "annual report", "financial year", "fiscal", "quarter",
    "board", "auditor", "governance", "esg", "sustainability",
    "acquisition", "subsidiary", "shareholders", "balance sheet",
]

# Used by Check 2: complete questions phrased the way a real user would type them.
#
# These are not keywords. They are examples of MEANING, and Check 2 measures how
# close an incoming question sits to them. Two rules when writing them:
#
#   1. Cover the breadth of the domain. Whatever area is missing here is the
#      area the guardrail will wrongly reject.
#   2. Include the comparison phrasing prominently, since year-over-year
#      questions are what this system is actually built for. If every example
#      were a single-year lookup, comparison questions would score lower than
#      they should.
DOMAIN_REFERENCE_EXAMPLES = [
    "How did revenue change between the two financial years?",
    "What did the company say about operating margins?",
    "What are the main risks disclosed in the annual report?",
    "How did the company describe its currency exposure?",
    "What changed in the risk factors compared with last year?",
    "How many employees does the company have?",
    "What was said about attrition and hiring?",
    "How did the segment performance differ across the years?",
    "What does management say about the outlook for next year?",
    "How did the company describe its AI strategy?",
    "What were the largest expenses during the year?",
    "How did the auditor's report describe the accounts?",
    "What acquisitions were completed during the year?",
    "What did the board report on governance?",
    "How did client concentration change year on year?",
    "What did the company disclose about its dividend?",
]


# ==============================================================================
# SECTION 2 - THE DECISION OBJECT
# ==============================================================================
# A guardrail should never return a bare True or False. The moment it does, every
# piece of information about WHY it decided that is thrown away -- and that is
# exactly what you need when someone reports that their question was refused.
# ==============================================================================

ALLOW = "allow"
BLOCK = "block"


@dataclass
class TopicDecision:
    """
    The result of the topic check.

    verdict   ALLOW or BLOCK.
    check     Which check decided -- "keyword" or "similarity".
    reason    A short explanation, written for the logs rather than the user.
    score     The similarity number, when there is one. The keyword check does
              not produce a score, which is why this is Optional and defaults to
              None.

    @dataclass writes the __init__ method for us from the field list above, so
    creating one is just TopicDecision(verdict=..., check=..., reason=...).
    """
    verdict: str
    check: str
    reason: str
    score: Optional[float] = None


# ==============================================================================
# SECTION 3 - CHECK 1: KEYWORD MATCHING
# ==============================================================================

def check_by_keywords(question: str) -> Optional[TopicDecision]:
    """
    Look for domain vocabulary in the question.

    Returns a decision when a keyword is found, or None when nothing is found.

    None means "no opinion", NOT "off topic". That distinction is the whole
    reason this function can be trusted as a fast path. A question like "how did
    things change for the company this year?" is clearly in scope and contains
    none of our words -- blocking it here would be wrong.

    ABOUT THE WORD BOUNDARY
    -----------------------
    \\b marks the edge of a word. Without it, "risk" would match inside "brisk"
    and "orders" inside "recorders". Small detail, real bugs.

    re.escape is a safety habit: if a keyword ever contained a character with
    special meaning in a regular expression, this makes sure it is treated as
    plain text instead.
    """
    normalised = question.lower()

    matched = []
    for keyword in DOMAIN_KEYWORDS:
        if re.search(rf"\b{re.escape(keyword)}\b", normalised):
            matched.append(keyword)

    if matched:
        return TopicDecision(
            verdict=ALLOW,
            check="keyword",
            reason=f"found domain terms: {', '.join(matched)}",
        )

    return None


# ==============================================================================
# SECTION 4 - CHECK 2: SEMANTIC SIMILARITY
# ==============================================================================
# Check 1 compares spellings. Check 2 compares meanings.
#
# We turn each reference example into an embedding, turn the question into an
# embedding, and measure how close the question sits to the nearest example.
#
# We take the MAXIMUM similarity rather than the average, deliberately. Our
# domain has separate areas -- risks, margins, headcount, governance -- and a
# question about governance is genuinely unlike a question about margins.
# Averaging would penalise every specific question for failing to resemble the
# entire domain at once.
# ==============================================================================

def embed_texts(texts: List[str]) -> np.ndarray:
    """
    Turn a list of texts into embeddings, sent in a single API call.

    Returns a table of numbers with one row per text.
    """
    response = client.embeddings.create(
        model=config.EMBEDDING_MODEL,
        input=texts,
    )
    return np.array([item.embedding for item in response.data])


def cosine_similarity(vector: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """
    Compare one embedding against every row of a table of embeddings.

    Returns one score per row: close to 1.0 for similar meaning, close to 0.0 for
    unrelated meaning.

    WHY COSINE
    ----------
    An embedding can be pictured as an arrow pointing somewhere. Cosine
    similarity measures the ANGLE between two arrows and ignores their length.

    That is what we want here. A short question and a long, rambling question
    about the same subject should count as similar -- their arrows point the same
    way even though one is longer, and it is the direction that carries meaning.

    The steps:
      - dividing by np.linalg.norm gives every arrow the same length, leaving
        only direction
      - the @ symbol is matrix multiplication, comparing our one question against
        all reference rows in a single operation
      - the tiny 1e-10 prevents a division by zero if an embedding were ever all
        zeros
    """
    vector_norm = vector / (np.linalg.norm(vector) + 1e-10)
    matrix_norms = matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-10)
    return matrix_norms @ vector_norm


# The reference embeddings never change while the program runs, so we work them
# out once and keep them here. Without this, every question would re-embed all 16
# reference examples, multiplying the embedding cost by 16 for no benefit.
_reference_embeddings: Optional[np.ndarray] = None


def get_reference_embeddings() -> np.ndarray:
    """
    Return the reference embeddings, computing them on first use.

    The "global" keyword tells Python we intend to change the variable defined
    outside this function. Without it, the assignment would create a new local
    variable that vanishes when the function ends, and the embeddings would be
    recomputed on every single call.
    """
    global _reference_embeddings

    if _reference_embeddings is None:
        _reference_embeddings = embed_texts(DOMAIN_REFERENCE_EXAMPLES)

    return _reference_embeddings


def check_by_similarity(question: str) -> TopicDecision:
    """
    Judge whether a question is in scope by comparing its meaning to the domain.

    Unlike Check 1, this always returns a decision. It is the final word.

    np.max gives the highest score. np.argmax gives the POSITION of that score,
    which we use to look up which reference example was closest -- useful in the
    logs when explaining why something was allowed or refused.
    """
    question_embedding = embed_texts([question])[0]
    scores = cosine_similarity(question_embedding, get_reference_embeddings())

    best_score = float(np.max(scores))
    closest_example = DOMAIN_REFERENCE_EXAMPLES[int(np.argmax(scores))]

    if best_score >= config.TOPIC_SIMILARITY_THRESHOLD:
        return TopicDecision(
            verdict=ALLOW,
            check="similarity",
            reason=f"close in meaning to: '{closest_example}'",
            score=best_score,
        )

    return TopicDecision(
        verdict=BLOCK,
        check="similarity",
        reason="not close to anything in the domain",
        score=best_score,
    )


# ==============================================================================
# SECTION 5 - PUTTING IT TOGETHER
# ==============================================================================

REFUSAL_MESSAGE = (
    "I can only answer questions about the annual reports of "
    + " and ".join(config.COMPANIES)
    + " -- things like revenue, margins, risks, strategy and how those changed "
    "between financial years. I can't give investment advice or help with other "
    "topics."
)


def check_topic(question: str) -> TopicDecision:
    """
    Run the full topic guardrail.

        empty question   ->  block immediately
        keyword found    ->  allow, with no API call at all
        otherwise        ->  let the similarity check decide
    """
    if not question.strip():
        return TopicDecision(BLOCK, "input", "question is empty")

    keyword_decision = check_by_keywords(question)
    if keyword_decision is not None:
        return keyword_decision

    return check_by_similarity(question)


# ==============================================================================
# SECTION 6 - RUNNING THIS FILE DIRECTLY
# ==============================================================================

SAMPLE_QUESTIONS = [
    # Obvious domain vocabulary -- Check 1 handles these with no API call.
    "How did revenue change between FY2025 and FY2026?",
    "What are the main risks disclosed in the report?",

    # Clearly in scope, but containing none of our keywords. Check 1 has nothing
    # to say; Check 2 recognises the meaning.
    "What did management say has changed for the business this year?",
    "How many people work at the company now compared to before?",

    # Nothing to do with us.
    "What is a good recipe for paneer butter masala?",
    "Write me a Python script to read a CSV file.",

    # Financial and about the right company, but asking for advice rather than
    # for what the report says. This is the boundary worth watching.
    "Should I buy shares in this company?",
    "Is this a good stock to hold for five years?",
]


if __name__ == "__main__":
    print("=" * 74)
    print("TOPIC GUARDRAIL")
    print("=" * 74)
    print()

    for question in SAMPLE_QUESTIONS:
        decision = check_topic(question)

        score_text = f"{decision.score:.3f}" if decision.score is not None else "  -  "
        print(f"  {decision.verdict.upper():5}  {score_text}  via {decision.check}")
        print(f"         {question}")
        print(f"         {decision.reason}")
        print()

    print("-" * 74)
    print("WHAT THE USER SEES WHEN BLOCKED")
    print("-" * 74)
    print()
    print(f"  {REFUSAL_MESSAGE}")
    print()
    print("  Note what the refusal does NOT do: it does not say which check")
    print("  fired or which words caused the problem. Explaining the rule is")
    print("  effectively explaining how to get around it. The detail belongs in")
    print("  the logs, where we need it.")
    print()