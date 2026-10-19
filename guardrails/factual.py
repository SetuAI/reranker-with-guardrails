"""
================================================================================
OUTPUT GUARDRAIL - FACTUAL GROUNDING
================================================================================

WHAT THIS FILE DOES
-------------------
Checks whether the answer the model wrote is actually supported by the chunks we
retrieved.

This runs AFTER generation and BEFORE the user sees anything. It is the last gate
in the pipeline, and it is the one that decides whether a human gets involved.

WHY "IS THIS TRUE" IS THE WRONG QUESTION
----------------------------------------
It is worth being honest about the difficulty here.

The other guardrails check properties we can define. Is this on topic? Does this
contain a PAN number? Does the answer mention both years? Each has a definite
answer.

"Is this true?" has no such shortcut. Verifying an arbitrary claim is about as
hard as producing it in the first place, so asking one model to fact-check
another is not obviously an improvement.

SO WE CHANGE THE QUESTION
-------------------------
Instead of asking "is this true?", which we cannot answer, we ask something we
can:

    Is everything in this answer supported by the chunks we retrieved?

That is narrower and has a definite answer. We are no longer judging truth in
general -- we are checking that the model stayed inside its sources.

This is called GROUNDING. It will not catch a wrong figure that was wrong in the
annual report itself. It catches the model inventing a figure that was never in
any retrieved chunk, which is the failure that actually happens.

WHY THIS MATTERS PARTICULARLY IN THIS PROJECT
---------------------------------------------
Our whole use case is comparison: what changed between FY2025 and FY2026.

Comparison is exactly where models invent things. Asked what changed, a model
will produce a fluent, confident narrative about shifting strategy and evolving
risk language, and some of that narrative will be assembled from what such
reports usually say rather than from what these two reports actually said.

THE APPROACH
------------
    Check 1   Number verification   instant, free, no model call
    Check 2   Claim verification    one model call, catches invented statements

Check 1 is unusually good value. Most damaging inventions in a financial context
are numeric, and a number either appears in the retrieved text or it does not.
No judgement required.

Check 1 is also YEAR-AWARE. A number that appears somewhere in the sources is not
enough: if the answer says "in FY2026 revenue was X", then X must appear in the
FY2026 extracts. On this corpus that matters more than anything else, because
the most likely numeric mistake is not an invented figure -- it is a real figure
attached to the wrong year.

And Check 1 is never allowed to approve a comparison on its own. "Revenue rose"
and "revenue fell" contain the same numbers; only reading the claim can tell them
apart. So any answer that uses comparison language always goes on to Check 2.

THE ACTION THIS GUARDRAIL TAKES IS DIFFERENT
--------------------------------------------
The topic guardrail blocks. This one ESCALATES.

When an answer looks ungrounded, discarding it outright is too blunt -- it may
well be correct and simply phrased in a way our checks could not follow. So
instead of throwing it away, we route it to a person.

That is the human-in-the-loop trigger for this entire project, and it is why this
file matters beyond its own checks. Everything the graph does with interrupts
starts from the verdict this file returns.

HOW TO RUN
----------
    python guardrails/factual.py

Runs on built-in examples and needs no ingested data.

================================================================================
"""

import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

from openai import OpenAI

sys.path.append(str(Path(__file__).parent.parent))

import config


client = OpenAI(api_key=config.OPENAI_API_KEY)


# ==============================================================================
# SECTION 1 - THE DECISION OBJECT
# ==============================================================================

GROUNDED = "grounded"            # Supported by the sources. Show it.
UNGROUNDED = "ungrounded"        # Invents or contradicts. Do not show it.
NEEDS_REVIEW = "needs_review"    # Unclear. Send it to a person.


@dataclass
class GroundingDecision:
    """
    The result of checking one answer against its sources.

    verdict            GROUNDED, UNGROUNDED or NEEDS_REVIEW.
    check              Which check decided -- "numbers" or "claims".
    reason             A short explanation for the logs.
    score              What proportion of numbers were traceable, where measured.
    unsupported_items  The specific numbers or claims we could not trace.

    THE LAST FIELD IS THE IMPORTANT ONE
    -----------------------------------
    A reviewer told "something in this answer is unsupported" has to re-read
    everything. A reviewer told "the figure 15.6% does not appear in any
    retrieved chunk" knows exactly where to look and can decide in seconds.

    The difference between those two experiences is the difference between a
    review step people use and one they quietly stop using.

    ABOUT field(default_factory=list)
    ---------------------------------
    We cannot write "unsupported_items: List[str] = []". In Python a default
    value is created ONCE, when the class is defined -- so every object would
    share the same list, and appending to one would change them all.

    default_factory=list tells the dataclass to call list() afresh for each new
    object. This is a classic Python trap and worth recognising on sight.
    """
    verdict: str
    check: str
    reason: str
    score: float = 0.0
    unsupported_items: List[str] = field(default_factory=list)


# ==============================================================================
# SECTION 2 - CHECK 1: NUMBER VERIFICATION
# ==============================================================================
# Pull every number out of the answer, pull every number out of the retrieved
# chunks, and see whether the first set is contained in the second.
#
# Crude, and far more useful than it sounds. In a financial document the numbers
# ARE the substance, and an invented number is the failure that causes real harm.
# This finds them in under a millisecond with no API call.
#
# It cannot catch a purely verbal invention -- "the company sounded more cautious
# about hiring" contains no numbers at all. That is Check 2's job.
# ==============================================================================

def extract_numbers(text: str) -> List[str]:
    """
    Find every number in a piece of text.

    ABOUT THE PATTERN
    -----------------
    \\d{1,3}(?:,\\d{2,3})+   numbers written with separators: 5,000 or 1,00,000
                            (Indian grouping puts 2 digits after the first group,
                            which is why it is {2,3} rather than {3})
    \\d+\\.?\\d*              plain numbers, with or without a decimal point
    |                       try the first alternative, then the second

    The comma pattern must come FIRST. If the plain pattern ran first it would
    match just "5" out of "5,000" and stop there, and we would end up comparing
    the wrong value entirely.

    Stripping the commas afterwards means "5,000" in the answer and "5000" in the
    source are recognised as the same number.
    """
    raw = re.findall(r"\d{1,3}(?:,\d{2,3})+|\d+\.?\d*", text)
    return [value.replace(",", "") for value in raw]


# Labels the answer may use for each fiscal year. "FY2025", "FY 2025",
# "FY25" and "2024-25" all refer to the year ended 31 March 2025.
def year_labels(fiscal_year: str) -> List[str]:
    """Return the regex patterns that name one fiscal year, e.g. FY2025."""
    end = int(fiscal_year[2:])
    start = end - 1
    return [
        rf"\bFY\s?-?\s?{end}\b",
        rf"\bFY\s?-?\s?{str(end)[2:]}\b",
        rf"\b{start}\s?[-–/]\s?{str(end)[2:]}\b",
    ]


def years_mentioned(sentence: str, fiscal_years: List[str]) -> List[str]:
    """Return which of the fiscal years a sentence refers to by name."""
    return [
        year for year in fiscal_years
        if any(re.search(p, sentence, re.IGNORECASE) for p in year_labels(year))
    ]


def strip_year_labels(sentence: str, fiscal_years: List[str]) -> str:
    """
    Remove the year labels themselves before extracting numbers.

    Otherwise "FY2026" contributes the number 2026, which is a label, not a
    figure, and would be checked against the sources like any other number.
    """
    for year in fiscal_years:
        for pattern in year_labels(year):
            sentence = re.sub(pattern, " ", sentence, flags=re.IGNORECASE)
    return sentence


def split_sentences(text: str) -> List[str]:
    """
    Split an answer into sentences and bullet lines.

    The split must not break decimals ("21.1") or amounts ("1,53,670"), so we
    only split on a full stop followed by whitespace, or on a line break.
    """
    parts = re.split(r"(?<=[.!?])\s+|\n+", text)
    return [part for part in parts if part.strip()]


def check_numbers(answer: str, sources: Dict[str, List[str]]) -> GroundingDecision:
    """
    Verify that the numbers in the answer appear in the right year's sources.

    sources maps each fiscal year to its chunk texts, e.g.
        {"FY2025": [...], "FY2026": [...]}

    THE YEAR RULE
    -------------
    Each sentence of the answer is checked on its own:

        names exactly one year   ->  its numbers must appear in THAT year's chunks
        names both or neither    ->  its numbers may appear in any chunk

    The answer is generated under "what FY2025 said" / "what FY2026 said"
    headings, so a heading line sets the year for the lines beneath it until the
    next heading names a different year.

    Sentences naming both years are usually the "what changed" summary, which
    legitimately quotes figures from each side, so they fall back to the pooled
    check rather than being rejected.

    ABOUT set()
    -----------
    We collect the source numbers into a set rather than a list. Checking whether
    an item is in a set is near-instant regardless of size, while checking a list
    means scanning it item by item.

    With five chunks the difference is invisible. With a long answer checked
    against many chunks it is the difference between instant and noticeable.
    """
    fiscal_years = list(sources)

    numbers_by_year = {}
    for year, texts in sources.items():
        numbers_by_year[year] = set()
        for source_text in texts:
            numbers_by_year[year].update(extract_numbers(source_text))
    all_source_numbers = set().union(*numbers_by_year.values())

    # Walk the answer sentence by sentence, remembering the year set by the most
    # recent heading so the lines beneath it inherit it.
    answer_numbers = []
    unsupported = []
    current_year = None
    for sentence in split_sentences(answer):
        named = years_mentioned(sentence, fiscal_years)
        if len(named) == 1:
            current_year = named[0]
            year = current_year
        elif len(named) > 1:
            current_year = None
            year = None
        else:
            year = current_year

        allowed = numbers_by_year[year] if year else all_source_numbers
        for number in extract_numbers(strip_year_labels(sentence, fiscal_years)):
            answer_numbers.append(number)
            if number not in allowed:
                if year and number in all_source_numbers:
                    unsupported.append(f"{number} (attributed to {year}, "
                                       f"found only in another year)")
                else:
                    unsupported.append(number)

    # An answer with no numbers cannot be judged by this check at all. We return
    # NEEDS_REVIEW rather than GROUNDED, which sends it on to Check 2.
    #
    # This case matters more here than in most projects. Comparison answers are
    # frequently entirely verbal -- "the company expanded its discussion of AI
    # risk" -- and those are exactly the answers most likely to be invented.
    if not answer_numbers:
        return GroundingDecision(
            verdict=NEEDS_REVIEW,
            check="numbers",
            reason="answer contains no numbers, this check cannot judge it",
        )

    supported_count = len(answer_numbers) - len(unsupported)
    ratio = supported_count / len(answer_numbers)

    if ratio >= config.GROUNDING_THRESHOLD:
        return GroundingDecision(
            verdict=GROUNDED,
            check="numbers",
            reason=f"{supported_count} of {len(answer_numbers)} numbers "
                   f"found in the retrieved text",
            score=ratio,
        )

    return GroundingDecision(
        verdict=NEEDS_REVIEW,
        check="numbers",
        reason=f"only {supported_count} of {len(answer_numbers)} numbers "
               f"found in the retrieved text",
        score=ratio,
        unsupported_items=unsupported,
    )


# ==============================================================================
# SECTION 3 - CHECK 2: CLAIM VERIFICATION
# ==============================================================================
# Hand the model the answer and the retrieved chunks together, and ask whether
# every statement can be traced to them.
#
# Two things in the prompt carry the weight:
#
#   1. The model is told to judge SUPPORT, not TRUTH. This is the whole idea of
#      grounding. A statement can be perfectly true and still ungrounded -- and
#      ungrounded is what matters to us, because it means the model wrote it from
#      general knowledge rather than from the documents in front of it.
#
#   2. The model must NAME the unsupported statement, not just report a verdict.
#      "Something here is unsupported" is useless to a reviewer.
# ==============================================================================

GROUNDING_PROMPT = """You check whether an answer is supported by the source \
text provided.

You are NOT judging whether the answer is true in general. You are judging one
thing only: can every statement in the answer be traced to the sources below?

A statement is UNSUPPORTED if it says something the sources do not say, even if
it sounds plausible or is probably correct.

Pay particular attention to comparisons. A claim that something increased,
decreased, strengthened or softened between two years is only supported if the
sources actually show both sides of that comparison, and the direction stated
matches them.

The sources are grouped by fiscal year. A statement about one year is only
supported by that year's sources -- a figure from FY2025 described as FY2026 is
UNSUPPORTED.

Reply in exactly this format:

VERDICT: SUPPORTED
or
VERDICT: UNSUPPORTED
UNSUPPORTED_CLAIM: <the specific statement not found in the sources>

SOURCES:
{sources}

ANSWER TO CHECK:
{answer}"""


def format_sources(sources: Dict[str, List[str]]) -> str:
    """
    Lay out the sources under a heading per year.

    The claim checker can only judge "FY2026 said X" if it can see which extract
    came from which year. Pooling them into one anonymous list would hide exactly
    the distinction this project is about.
    """
    sections = []
    for year, texts in sources.items():
        body = "\n\n---\n\n".join(texts) if texts else "(no extracts)"
        sections.append(f"=== {year} ===\n{body}")
    return "\n\n".join(sections)


def check_claims(answer: str, sources: Dict[str, List[str]]) -> GroundingDecision:
    """
    Ask a model whether the answer is supported by the sources.

    Temperature is 0 so the same answer always produces the same verdict. A check
    that answers differently to identical input cannot be tested and produces
    complaints nobody can reproduce.

    ON FAILURE WE RETURN NEEDS_REVIEW, NOT UNGROUNDED
    -------------------------------------------------
    This is a deliberate difference from a safety guardrail, where an API failure
    means blocking outright.

    The reasoning: a safety failure risks real harm, so refusing is right. A
    grounding failure usually means an answer is slightly loose. Discarding every
    answer whenever a network call times out would break the product for no
    safety benefit. Handing it to a person is the proportionate response.

    "Always fail closed" is too blunt a rule to apply everywhere without thinking
    about what the failure actually costs.
    """
    combined_sources = format_sources(sources)

    try:
        response = client.chat.completions.create(
            model=config.GUARDRAIL_MODEL,
            temperature=0,
            max_tokens=150,
            messages=[{
                "role": "user",
                "content": GROUNDING_PROMPT.format(
                    sources=combined_sources,
                    answer=answer,
                ),
            }],
        )

        reply = response.choices[0].message.content.strip()

        if "VERDICT: SUPPORTED" in reply.upper():
            return GroundingDecision(
                verdict=GROUNDED,
                check="claims",
                reason="every statement traced to the retrieved text",
                score=1.0,
            )

        # Pull out the specific claim the model flagged, if it gave one.
        match = re.search(r"UNSUPPORTED_CLAIM:\s*(.+)", reply, re.IGNORECASE)
        claim = match.group(1).strip() if match else "not specified"

        return GroundingDecision(
            verdict=NEEDS_REVIEW,
            check="claims",
            reason="a statement could not be traced to the retrieved text",
            score=0.0,
            unsupported_items=[claim],
        )

    except Exception as error:
        return GroundingDecision(
            verdict=NEEDS_REVIEW,
            check="claims",
            reason=f"grounding check unavailable ({type(error).__name__})",
        )


# ==============================================================================
# SECTION 4 - PUTTING IT TOGETHER
# ==============================================================================

# Words that assert a change between years. An answer using any of them makes a
# claim about direction that matching numbers alone cannot verify.
COMPARISON_PATTERN = re.compile(
    r"\b(increas|decreas|ris(e|es|ing)|rose|fell|fall|declin|grew|grow|"
    r"improv|worsen|deteriorat|expand|contract|reduc|higher|lower|up from|"
    r"down from|compared|versus|vs\.?|change|strengthen|weaken|soften)",
    re.IGNORECASE,
)


def check_grounding(answer: str, sources: Dict[str, List[str]]) -> GroundingDecision:
    """
    Run the full grounding guardrail.

        numbers fail                     ->  ask the model to check the claims
        numbers pass, answer compares    ->  ask the model anyway
        numbers pass, no comparison      ->  grounded, no API call needed

    Check 1 works as a filter. It can reject cheaply, but it can only ACCEPT an
    answer that makes no claim about direction. "Margin rose from 20 to 21" and
    "margin fell from 21 to 20" contain the same numbers, so for any comparison
    the model has to read the words.

    In a project about comparison, that means most answers reach Check 2. That
    is the honest cost of checking comparisons at all.
    """
    if not answer.strip():
        return GroundingDecision(UNGROUNDED, "input", "answer is empty")

    number_decision = check_numbers(answer, sources)

    if number_decision.verdict == GROUNDED and not COMPARISON_PATTERN.search(answer):
        return number_decision

    if number_decision.unsupported_items:
        # A number that does not trace back still needs a person, even if the
        # claim checker is satisfied -- the model reading prose can miss a digit.
        # Keep Check 1's specific findings: "17.4 attributed to FY2026, found
        # only in another year" is more useful to a reviewer than anything vaguer.
        claim_decision = check_claims(answer, sources)
        if claim_decision.verdict == GROUNDED:
            return number_decision
        claim_decision.unsupported_items = (
            number_decision.unsupported_items + claim_decision.unsupported_items
        )
        return claim_decision

    return check_claims(answer, sources)


def collect_source_texts(reranked_chunks: Dict[str, List[Dict]]) -> Dict[str, List[str]]:
    """
    Reduce the per-year chunks to their texts, KEEPING the split by year.

    An earlier version flattened everything into one list on the grounds that
    this check only needed to know whether a statement appeared anywhere. That
    was wrong for this corpus: a figure that appears "anywhere" may well be the
    other year's figure, and that is the mistake most worth catching here.
    """
    return {
        year: [result["text"] for result in year_results]
        for year, year_results in reranked_chunks.items()
    }


# ==============================================================================
# SECTION 5 - RUNNING THIS FILE DIRECTLY
# ==============================================================================

SAMPLE_SOURCES = {
    "FY2025": [
        """Revenue for the year stood at 1,53,670 crore, an increase of 6.1 per cent
        over the previous year. Operating margin was 21.1 per cent.""",

        """The Company operates internationally and is exposed to foreign exchange
        risk, principally in US dollars, euro and pound sterling. Attrition for the
        year was 14.2 per cent.""",
    ],
    "FY2026": [
        """Revenue for the year stood at 1,62,990 crore. Operating margin was
        20.8 per cent. Attrition for the year was 13.9 per cent.""",
    ],
}

SAMPLE_ANSWERS = [
    # Every number appears in the right year, and no change is claimed. Check 1
    # accepts it with no API call.
    "In FY2025 revenue was 1,53,670 crore with an operating margin of 21.1 per cent.",

    # A real figure attached to the wrong year: 21.1 is FY2025's margin. A
    # pooled check would have accepted this. The year-aware check does not.
    "In FY2026 the operating margin was 21.1 per cent.",

    # An invented margin: 24.8 appears in neither year.
    "The operating margin improved to 24.8 per cent during the year.",

    # No numbers at all, so Check 1 cannot judge it and passes it on. The claim
    # about a strategic shift appears nowhere in the sources.
    "The company shifted its strategy towards domestic markets and reduced its "
    "reliance on overseas clients.",

    # Also numberless, but this one IS supported by the sources.
    "The company is exposed to foreign exchange risk, mainly in US dollars, "
    "euro and pound sterling.",
]


if __name__ == "__main__":
    print("=" * 74)
    print("FACTUAL GROUNDING GUARDRAIL")
    print("=" * 74)

    print()
    print("-" * 74)
    print("CHECK 1 ALONE - NUMBER VERIFICATION (no API calls)")
    print("-" * 74)
    print()

    for answer in SAMPLE_ANSWERS:
        decision = check_numbers(answer, SAMPLE_SOURCES)
        print(f"  {decision.verdict.upper():13} {answer[:56]}")
        print(f"                {decision.reason}")
        if decision.unsupported_items:
            print(f"                not in sources: "
                  f"{', '.join(decision.unsupported_items)}")
        print()

    print("-" * 74)
    print("BOTH CHECKS TOGETHER")
    print("-" * 74)
    print()

    for answer in SAMPLE_ANSWERS:
        decision = check_grounding(answer, SAMPLE_SOURCES)
        print(f"  {decision.verdict.upper():13} via {decision.check}")
        print(f"                {answer[:56]}")
        print(f"                {decision.reason}")
        if decision.unsupported_items:
            print(f"                flagged: {decision.unsupported_items[0][:52]}")
        print()

    print("  A NEEDS_REVIEW verdict is what stops the graph and brings in a")
    print("  person. It is not a failure of the pipeline -- it is the pipeline")
    print("  correctly reporting that it is not confident enough to answer alone.")
    print()