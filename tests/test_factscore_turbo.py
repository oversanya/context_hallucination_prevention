"""
Unit tests for FActScoreTurbo.

All LLM calls are monkeypatched — no Ollama instance required.
"""
from __future__ import annotations

import math

import pytest

from src.factscore_turbo import FActScoreTurbo, FActScoreResult


# ─── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture()
def scorer() -> FActScoreTurbo:
    """FActScoreTurbo instance with a dummy model name (Ollama not called)."""
    return FActScoreTurbo(model="test-model", batch_verify=True, max_retries=0)


# ─── Tests ────────────────────────────────────────────────────────────────────

def test_decompose_returns_list(scorer: FActScoreTurbo, monkeypatch: pytest.MonkeyPatch) -> None:
    """decompose() must return a non-empty list when the LLM yields fact lines."""
    monkeypatch.setattr(
        scorer,
        "_chat",
        lambda system, user, max_tokens=512: "The sky is blue.\nWater is H2O.\n",
    )
    facts = scorer.decompose("The sky is blue and water is H2O.")
    assert isinstance(facts, list)
    assert len(facts) >= 1


def test_score_no_facts_returns_1(scorer: FActScoreTurbo, monkeypatch: pytest.MonkeyPatch) -> None:
    """score() must return 1.0 when decompose extracts no verifiable facts."""
    monkeypatch.setattr(scorer, "_chat", lambda *args, **kwargs: "NO_FACTS")
    result = scorer.score(response="Hmm, okay.", context="Some context.")
    assert result.score == 1.0
    assert result.n_facts == 0
    assert result.n_supported == 0
    assert result.error is None


def test_verify_batch_fallback(scorer: FActScoreTurbo, monkeypatch: pytest.MonkeyPatch) -> None:
    """
    _verify_batch must fall back to single verification when JSON parse fails.
    Verifies that the fallback path returns a list of bools of the correct length.
    """
    # Use claims that are unambiguously distinct from the context text.
    facts = ["Alpha statement is correct.", "Beta statement is incorrect."]
    call_log: list[str] = []

    def fake_chat(system: str, user: str, max_tokens: int = 512) -> str:
        call_log.append(user)
        if "JSON array" in user:
            # Simulate malformed JSON — triggers fallback to single verification.
            return "I cannot produce JSON."
        # Single-fact fallback: route by claim keyword.
        if "Alpha statement" in user:
            return "SUPPORTED"
        return "NOT_SUPPORTED"

    monkeypatch.setattr(scorer, "_chat", fake_chat)
    # Context intentionally contains neither "Alpha" nor "Beta" to avoid routing issues.
    results = scorer._verify_batch(facts, context="The quick brown fox jumps.")
    assert len(results) == 2
    assert results[0] is True
    assert results[1] is False
    # Confirm fallback was triggered (batch call + 2 single calls = 3 total).
    assert len(call_log) == 3


def test_result_is_hallucinated(scorer: FActScoreTurbo, monkeypatch: pytest.MonkeyPatch) -> None:
    """is_hallucinated() threshold boundary: score == threshold → not hallucinated."""
    # Decompose returns two facts long enough to pass the len>12 filter,
    # triggering batch_verify (requires >1 fact).
    DECOMPOSE_REPLY = "Paris is the capital of France.\nThe Eiffel Tower is in Paris."

    def fake_chat(system: str, user: str, max_tokens: int = 512) -> str:
        if "JSON array" in user:
            # Batch verify: first supported, second not — score = 0.5.
            return "[true, false]"
        return DECOMPOSE_REPLY

    monkeypatch.setattr(scorer, "_chat", fake_chat)
    result = scorer.score(
        response="Paris is the capital of France. The Eiffel Tower is in Paris.",
        context="Paris is the capital of France.",
    )
    # score = 1/2 = 0.5 → exactly at default threshold → NOT hallucinated.
    assert result.score == pytest.approx(0.5)
    assert result.n_facts == 2
    assert result.is_hallucinated(threshold=0.5) is False
    # One step below threshold → hallucinated.
    low_result = FActScoreResult(score=0.49, n_facts=2, n_supported=0)
    assert low_result.is_hallucinated(threshold=0.5) is True
