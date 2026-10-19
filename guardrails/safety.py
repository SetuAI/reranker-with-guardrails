"""
================================================================================
INPUT GUARDRAIL - SAFETY
================================================================================

WHAT THIS FILE DOES
-------------------
Checks the question for two things we never want to pass along, before the topic
check and before anything that costs money:

    1. Personal data        a PAN, an Aadhaar number, an email address or a phone
                            number typed into the question
    2. Investment advice    "should I buy Infosys shares?"

WHY THESE TWO, AND WHY HERE
---------------------------
Personal data has no business in a question about annual reports. If it is
there, the user pasted it by mistake -- and every later step would copy it
somewhere: into the embedding API, into the generation prompt, into the logs.
Stopping it at the very first step is the only place where stopping it means it
went nowhere.

Investment advice is the request this system must refuse even though it is ON
topic. "Should I buy Infosys?" mentions the company and would sail through the
topic check. The generation prompt already tells the model not to advise, but a
prompt is a request, not a rule. A refusal here is a rule.

WHY PATTERNS AND NOT A MODEL
----------------------------
Both are properties with a definite shape. A PAN is five letters, four digits,
one letter. "Should I buy" is a phrase. Patterns decide these in microseconds,
for free, and the same way every time -- which is exactly what the first gate of
a pipeline should be.

The cost is that patterns are narrow. "Is now a good moment to load up on
Eternal?" slips past. That is an honest trade: this check catches the plain
cases cheaply, and the instruction in the generation prompt remains as a second
line behind it.

HOW TO RUN
----------
    python guardrails/safety.py

Runs on built-in examples. No API calls.

================================================================================
"""

import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

sys.path.append(str(Path(__file__).parent.parent))


# ==============================================================================
# SECTION 1 - THE DECISION OBJECT
# ==============================================================================

ALLOW = "allow"
BLOCK = "block"


@dataclass
class SafetyDecision:
    """
    The result of the safety check.

    verdict   ALLOW or BLOCK.
    check     Which check decided -- "personal_data" or "advice".
    reason    A short explanation for the logs. It names the KIND of data found,
              never the data itself -- writing the PAN into the log would undo
              the point of catching it.
    """
    verdict: str
    check: str
    reason: str


# ==============================================================================
# SECTION 2 - CHECK 1: PERSONAL DATA
# ==============================================================================
# Each entry is (name for the log, pattern).
#
# \b is a word boundary: it stops "ABCDE1234F" matching inside a longer code.
# ==============================================================================

PERSONAL_DATA_PATTERNS = [
    # PAN: five letters, four digits, one letter.
    ("PAN", re.compile(r"\b[A-Z]{5}\d{4}[A-Z]\b", re.IGNORECASE)),

    # Aadhaar: twelve digits, usually written in groups of four. It never starts
    # with 0 or 1, which keeps it from matching ordinary large numbers.
    ("Aadhaar number", re.compile(r"\b[2-9]\d{3}\s?\d{4}\s?\d{4}\b")),

    ("email address", re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")),

    # Indian mobile: ten digits starting 6-9, optionally with +91.
    ("phone number", re.compile(r"(?:\+91[\s-]?)?\b[6-9]\d{9}\b")),
]


def check_personal_data(question: str) -> Optional[SafetyDecision]:
    """Block the question if it contains personal data. None means none found."""
    for name, pattern in PERSONAL_DATA_PATTERNS:
        if pattern.search(question):
            return SafetyDecision(BLOCK, "personal_data", f"question contains personal data ({name})")
    return None


# ==============================================================================
# SECTION 3 - CHECK 2: INVESTMENT ADVICE
# ==============================================================================
# Phrases that ask for a recommendation rather than for what the report says.
#
# Note what is NOT here: "investment", "shares", "dividend". Those are perfectly
# good words in a question about an annual report -- "how did the dividend
# change?" is exactly what this system is for. Only the asking-for-advice
# phrasing is blocked.
# ==============================================================================

ADVICE_PATTERNS: List[re.Pattern] = [
    re.compile(p, re.IGNORECASE) for p in [
        r"\bshould (i|we) (buy|sell|invest|hold|short)\b",
        r"\b(is|are) \w+(\s\w+)? (a )?(good|bad|safe) (buy|investment|bet)\b",
        r"\bworth (buying|investing|selling)\b",
        r"\b(buy|sell|hold) (or|vs\.?) (buy|sell|hold)\b",
        r"\b(target price|price target|stock tip)\b",
        r"\bwhich (stock|share|company) should\b",
    ]
]

ADVICE_REFUSAL_MESSAGE = (
    "I can't give investment advice or say whether to buy or sell shares. I can "
    "tell you what the annual reports say -- for example how revenue, margins or "
    "risks changed between the two years."
)

PERSONAL_DATA_REFUSAL_MESSAGE = (
    "Your question seems to contain personal information (such as a PAN, Aadhaar "
    "number, email or phone number). Please remove it and ask again -- it isn't "
    "needed to answer questions about the annual reports."
)


def check_advice(question: str) -> Optional[SafetyDecision]:
    """Block requests for investment advice. None means none found."""
    for pattern in ADVICE_PATTERNS:
        if pattern.search(question):
            return SafetyDecision(BLOCK, "advice", "question asks for investment advice")
    return None


# ==============================================================================
# SECTION 4 - PUTTING IT TOGETHER
# ==============================================================================

def check_safety(question: str) -> SafetyDecision:
    """
    Run the full safety guardrail.

        personal data found   ->  block
        advice requested      ->  block
        otherwise             ->  allow, and let the topic check decide next

    Personal data is checked first: if a question has both, the user most needs
    to hear about the data they pasted.
    """
    return (
        check_personal_data(question)
        or check_advice(question)
        or SafetyDecision(ALLOW, "safety", "no personal data or advice request found")
    )


def refusal_message(decision: SafetyDecision) -> str:
    """The message the user sees for a blocked question."""
    if decision.check == "personal_data":
        return PERSONAL_DATA_REFUSAL_MESSAGE
    return ADVICE_REFUSAL_MESSAGE


# ==============================================================================
# SECTION 5 - RUNNING THIS FILE DIRECTLY
# ==============================================================================

SAMPLE_QUESTIONS = [
    "How did Infosys's revenue change between the two years?",
    "How did the dividend per share change?",
    "Should I buy Infosys shares now?",
    "Is Eternal a good investment for the long term?",
    "My PAN is ABCDE1234F, how did revenue change?",
    "Email me at someone@example.com about attrition",
]


if __name__ == "__main__":
    print("=" * 74)
    print("SAFETY GUARDRAIL")
    print("=" * 74)
    print()

    for question in SAMPLE_QUESTIONS:
        decision = check_safety(question)
        print(f"  {decision.verdict.upper():6} {question[:60]}")
        print(f"         {decision.reason}")
        print()
