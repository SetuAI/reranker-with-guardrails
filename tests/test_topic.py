"""
Fixed cases for the topic guardrail's routing.

The similarity check calls the embedding API, so it is replaced here. What these
tests pin down is the ORDER of the checks -- the cheap ones decide first, and the
API is only reached when they cannot.
"""

import pytest

from guardrails import topic
from guardrails.topic import ALLOW, BLOCK, TopicDecision, check_topic


@pytest.fixture
def similarity(monkeypatch):
    """Replace the embedding-based check with a stub that records each call."""
    calls = []

    def stub(question):
        calls.append(question)
        return TopicDecision(BLOCK, "similarity", "stub")

    monkeypatch.setattr(topic, "check_by_similarity", stub)
    return calls


def test_empty_question_is_blocked_without_an_api_call(similarity):
    assert check_topic("   ").verdict == BLOCK
    assert similarity == []


def test_domain_keyword_is_allowed_without_an_api_call(similarity):
    question = f"How did {topic.DOMAIN_KEYWORDS[0]} change between the two years?"
    assert check_topic(question).verdict == ALLOW
    assert similarity == []


def test_off_topic_question_reaches_the_similarity_check(similarity):
    decision = check_topic("Write me a poem about the sea")
    assert similarity == ["Write me a poem about the sea"]
    assert decision.verdict == BLOCK
