"""
Fixed cases for the two pattern-based guardrails: safety on the way in, format
on the way out. Neither makes an API call.
"""

import pytest

from guardrails.format import check_format
from guardrails.safety import ALLOW, BLOCK, check_safety


# ------------------------------------------------------------------------------
# Safety
# ------------------------------------------------------------------------------

@pytest.mark.parametrize("question", [
    "How did Infosys's revenue change between the two years?",
    "How did the dividend per share change?",
    "What investments did the company make in FY2026?",
    "Revenue was 1,53,670 crore -- how did it change?",
])
def test_ordinary_questions_are_allowed(question):
    assert check_safety(question).verdict == ALLOW


@pytest.mark.parametrize("question, kind", [
    ("My PAN is ABCDE1234F, how did revenue change?", "PAN"),
    ("Aadhaar 2345 6789 0123 -- what was attrition?", "Aadhaar"),
    ("Send it to someone@example.com", "email"),
    ("Call me on +91 9876543210 about margins", "phone"),
])
def test_personal_data_is_blocked(question, kind):
    decision = check_safety(question)
    assert decision.verdict == BLOCK
    assert kind in decision.reason


def test_the_log_reason_never_repeats_the_personal_data():
    decision = check_safety("My PAN is ABCDE1234F")
    assert "ABCDE1234F" not in decision.reason


@pytest.mark.parametrize("question", [
    "Should I buy Infosys shares now?",
    "Is it a good investment for the long term?",
    "Is Eternal worth buying?",
    "Is Eternal a good investment for the long term?",
    "What is your target price for Infosys?",
])
def test_advice_requests_are_blocked(question):
    assert check_safety(question).check == "advice"


# ------------------------------------------------------------------------------
# Format
# ------------------------------------------------------------------------------

def test_well_formed_comparison_passes():
    answer = ("In FY2025 revenue was 1,53,670 crore. In FY2026 it was "
              "1,62,990 crore, an increase between the two years.")
    assert check_format(answer).passed


def test_answer_covering_one_year_fails():
    decision = check_format("In FY2026 revenue was 1,62,990 crore, which increased.")
    assert not decision.passed
    assert decision.problems == ["answer does not mention FY2025"]


def test_advice_in_the_answer_fails():
    decision = check_format("Revenue increased in FY2026 compared with FY2025. "
                            "We recommend buying the shares.")
    assert "answer gives investment advice" in decision.problems


def test_every_broken_rule_is_reported():
    decision = check_format("Revenue was fine.")
    assert len(decision.problems) >= 3


def test_honest_cannot_answer_is_not_punished():
    assert check_format("The extracts do not contain enough information to "
                        "answer this question.").passed
