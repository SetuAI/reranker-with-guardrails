"""
Fixed cases for the grounding guardrail.

The point of these is not coverage for its own sake. Each one pins down a
behaviour the README promises, so that tuning a threshold or rewording a prompt
cannot quietly break it.
"""

import pytest

from guardrails import factual
from guardrails.factual import (
    GROUNDED,
    NEEDS_REVIEW,
    GroundingDecision,
    check_grounding,
    check_numbers,
    collect_source_texts,
)


SOURCES = {
    "FY2025": ["Revenue stood at 1,53,670 crore. Operating margin was 21.1 per cent."],
    "FY2026": ["Revenue stood at 1,62,990 crore. Operating margin was 20.8 per cent."],
}


@pytest.fixture
def claims(monkeypatch):
    """
    Replace the model-based claim check with a stub that records each call.

    Set claims.verdict to choose what the stub "model" decides.
    """
    class Stub:
        verdict = GROUNDED
        calls = 0

        def __call__(self, answer, sources):
            self.calls += 1
            items = [] if self.verdict == GROUNDED else ["a flagged claim"]
            return GroundingDecision(self.verdict, "claims", "stub", unsupported_items=items)

    stub = Stub()
    monkeypatch.setattr(factual, "check_claims", stub)
    return stub


# ------------------------------------------------------------------------------
# Check 1 on its own
# ------------------------------------------------------------------------------

def test_figure_in_its_own_year_passes():
    decision = check_numbers("In FY2025 revenue was 1,53,670 crore.", SOURCES)
    assert decision.verdict == GROUNDED


def test_figure_attributed_to_wrong_year_is_caught():
    # 21.1 is the FY2025 margin. A pooled check would accept it.
    decision = check_numbers("In FY2026 the operating margin was 21.1 per cent.", SOURCES)
    assert decision.verdict == NEEDS_REVIEW
    assert "attributed to FY2026" in decision.unsupported_items[0]


def test_heading_sets_the_year_for_the_lines_beneath_it():
    answer = (
        "What FY2025 said:\n"
        "- Margin was 21.1 per cent.\n"
        "What FY2026 said:\n"
        "- Margin was 21.1 per cent."
    )
    decision = check_numbers(answer, SOURCES)
    assert decision.verdict == NEEDS_REVIEW
    assert len(decision.unsupported_items) == 1


def test_sentence_naming_both_years_may_quote_either():
    answer = "Margin moved from 21.1 per cent in FY2025 to 20.8 per cent in FY2026."
    assert check_numbers(answer, SOURCES).verdict == GROUNDED


def test_alternative_year_labels_are_recognised():
    assert check_numbers("In 2025-26 revenue was 1,53,670 crore.", SOURCES).verdict == NEEDS_REVIEW
    assert check_numbers("In FY26 revenue was 1,62,990 crore.", SOURCES).verdict == GROUNDED


def test_invented_number_is_caught():
    decision = check_numbers("Operating margin was 24.8 per cent.", SOURCES)
    assert decision.verdict == NEEDS_REVIEW
    assert decision.unsupported_items == ["24.8"]


# ------------------------------------------------------------------------------
# The two checks together
# ------------------------------------------------------------------------------

def test_plain_statement_with_traced_numbers_needs_no_model_call(claims):
    decision = check_grounding("In FY2025 revenue was 1,53,670 crore.", SOURCES)
    assert decision.verdict == GROUNDED
    assert claims.calls == 0


def test_comparison_always_goes_to_the_claim_check(claims):
    # Every number traces back, but "rose" is wrong -- the margin fell.
    claims.verdict = NEEDS_REVIEW
    answer = "Operating margin rose from 21.1 per cent in FY2025 to 20.8 per cent in FY2026."
    decision = check_grounding(answer, SOURCES)
    assert claims.calls == 1
    assert decision.verdict == NEEDS_REVIEW


def test_numberless_answer_approved_by_claim_check_is_grounded(claims):
    decision = check_grounding("The company is exposed to foreign exchange risk.", SOURCES)
    assert claims.calls == 1
    assert decision.verdict == GROUNDED


def test_untraced_number_escalates_even_if_claim_check_passes(claims):
    decision = check_grounding("In FY2026 the operating margin was 21.1 per cent.", SOURCES)
    assert decision.verdict == NEEDS_REVIEW
    assert "attributed to FY2026" in decision.unsupported_items[0]


def test_findings_from_both_checks_reach_the_reviewer(claims):
    claims.verdict = NEEDS_REVIEW
    decision = check_grounding("Operating margin was 24.8 per cent.", SOURCES)
    assert decision.unsupported_items == ["24.8", "a flagged claim"]


def test_empty_answer_is_rejected_outright(claims):
    assert check_grounding("   ", SOURCES).verdict == factual.UNGROUNDED
    assert claims.calls == 0


def test_claim_check_failure_escalates_rather_than_blocks(monkeypatch):
    def broken(*args, **kwargs):
        raise TimeoutError

    monkeypatch.setattr(factual.client.chat.completions, "create", broken)
    decision = factual.check_claims("anything", SOURCES)
    assert decision.verdict == NEEDS_REVIEW


def test_collect_source_texts_keeps_years_apart():
    chunks = {"FY2025": [{"text": "a"}], "FY2026": [{"text": "b"}, {"text": "c"}]}
    assert collect_source_texts(chunks) == {"FY2025": ["a"], "FY2026": ["b", "c"]}
