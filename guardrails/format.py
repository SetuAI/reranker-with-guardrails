"""
================================================================================
OUTPUT GUARDRAIL - FORMAT
================================================================================

WHAT THIS FILE DOES
-------------------
Checks the SHAPE of the draft answer before the grounding check reads its
content:

    1. It covers both years      FY2025 and FY2026 are each named
    2. It says what changed      or says plainly that nothing changed
    3. It gives no advice        no "we recommend buying", "a good investment"
    4. It is a sensible length   not empty, not a runaway essay

WHY CHECK SHAPE SEPARATELY FROM GROUNDING
-----------------------------------------
Grounding asks "is every statement supported?". An answer can pass that and still
be wrong for this system: it can describe FY2026 perfectly and never mention
FY2025. Every statement is supported -- but it is not a comparison, which is the
one thing the user asked for.

Shape is also free to check. It runs before grounding, so a malformed answer goes
to review without spending a model call on it.

ONE FUNCTION PER RULE
---------------------
Each rule is a small function that returns a problem in plain English, or None
if the rule holds. check_format runs ALL of them and collects every problem --
not just the first. That list goes straight to the reviewer, so they see "answer
does not mention FY2025" rather than "format problem".

(Pydantic validators look like a natural fit here, but pydantic stops checking a
field at its first failed validator, so the reviewer would only ever see one
problem at a time.)

THE ACTION THIS GUARDRAIL TAKES
-------------------------------
Like grounding, it ESCALATES rather than blocks. A badly shaped answer may still
be usable after a person looks at it.

HOW TO RUN
----------
    python guardrails/format.py

Runs on built-in examples. No API calls.

================================================================================
"""

import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional

sys.path.append(str(Path(__file__).parent.parent))

from guardrails.factual import year_labels


# Limits on length, in characters. Below the minimum the model has not really
# answered; above the maximum it has usually drifted from the extracts.
MIN_ANSWER_LENGTH = 40
MAX_ANSWER_LENGTH = 6000

# Phrases that mean the model gave advice despite being told not to.
ADVICE_PHRASES = re.compile(
    r"\b(we|i) (would )?(recommend|suggest|advise) (buying|selling|holding|investing)|"
    r"\b(good|attractive|strong) (investment|buy)\b|\b(buy|sell) rating\b",
    re.IGNORECASE,
)

# Words that show the answer addresses change -- or says there was none.
CHANGE_WORDS = re.compile(
    r"\b(chang|increas|decreas|rose|fell|grew|declin|improv|differ|"
    r"same|similar|unchanged|consistent|no meaningful|remained)",
    re.IGNORECASE,
)

# Phrases the generation prompt tells the model to use when it cannot answer.
# Such an answer will not cover both years, and should not be punished for it.
CANNOT_ANSWER = re.compile(
    r"\b(do not|does not|don't|doesn't) (contain|provide|include|mention)|"
    r"\bnot enough information\b|\bno relevant extracts\b",
    re.IGNORECASE,
)


# ==============================================================================
# SECTION 1 - THE RULES
# ==============================================================================
# Each rule takes the answer and returns a problem, or None if all is well.
# ==============================================================================

def names_year(text: str, fiscal_year: str) -> bool:
    return any(re.search(p, text, re.IGNORECASE) for p in year_labels(fiscal_year))


def sensible_length(text: str) -> Optional[str]:
    if len(text.strip()) < MIN_ANSWER_LENGTH:
        return "answer is too short to be a real answer"
    if len(text) > MAX_ANSWER_LENGTH:
        return f"answer is longer than {MAX_ANSWER_LENGTH} characters"
    return None


def no_advice(text: str) -> Optional[str]:
    if ADVICE_PHRASES.search(text):
        return "answer gives investment advice"
    return None


def covers_both_years(text: str) -> Optional[str]:
    if CANNOT_ANSWER.search(text):
        return None
    missing = [year for year in ("FY2025", "FY2026") if not names_year(text, year)]
    if missing:
        return f"answer does not mention {' or '.join(missing)}"
    return None


def says_what_changed(text: str) -> Optional[str]:
    if CANNOT_ANSWER.search(text):
        return None
    if not CHANGE_WORDS.search(text):
        return "answer does not say what changed, or that nothing did"
    return None


RULES: List[Callable[[str], Optional[str]]] = [
    sensible_length,
    no_advice,
    covers_both_years,
    says_what_changed,
]


# ==============================================================================
# SECTION 2 - RUNNING THE CHECK
# ==============================================================================

@dataclass
class FormatDecision:
    """
    passed     True if every rule held.
    problems   One plain sentence per broken rule, for the reviewer.
    """
    passed: bool
    problems: List[str] = field(default_factory=list)


def check_format(answer: str) -> FormatDecision:
    """Check the answer against every rule and report all that failed."""
    problems = [problem for rule in RULES if (problem := rule(answer))]
    return FormatDecision(passed=not problems, problems=problems)


# ==============================================================================
# SECTION 3 - RUNNING THIS FILE DIRECTLY
# ==============================================================================

SAMPLE_ANSWERS = [
    "In FY2025 revenue was 1,53,670 crore. In FY2026 it was 1,62,990 crore, "
    "an increase between the two years.",

    "In FY2026 revenue was 1,62,990 crore, which increased from the prior year.",

    "Revenue increased in FY2026 compared with FY2025. We recommend buying the "
    "shares.",

    "The extracts do not contain enough information to answer this question.",
]


if __name__ == "__main__":
    print("=" * 74)
    print("FORMAT GUARDRAIL")
    print("=" * 74)
    print()

    for answer in SAMPLE_ANSWERS:
        decision = check_format(answer)
        print(f"  {'PASS' if decision.passed else 'FAIL':5} {answer[:62]}")
        for problem in decision.problems:
            print(f"        - {problem}")
        print()
