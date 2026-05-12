"""
Unit tests for src/judge.py.

All Ollama calls are intercepted via a fake client passed through the
``client`` keyword on each LLMJudge call site.
"""
from __future__ import annotations

from unittest.mock import patch

import pytest

from src.judge import (
    LLMJudge,
    PairwiseResult,
    _deterministic_swap,
    _parse_ab_json,
    _parse_likert_json,
)


# ─── Fake Ollama client ───────────────────────────────────────────────────────

class _FakeClient:
    """Stand-in for the ``ollama`` module: returns a fixed reply per model."""

    def __init__(self, replies: dict):
        # replies: {model_tag: list[str]}  (consumed in order)
        self._replies = {k: list(v) for k, v in replies.items()}
        self.calls: list[tuple] = []

    def chat(self, model: str, messages, options):  # noqa: D401 — mimics SDK
        self.calls.append((model, messages, options))
        if not self._replies.get(model):
            raise RuntimeError(f"no canned reply for model {model!r}")
        reply = self._replies[model].pop(0)
        return {"message": {"content": reply}}


# ─── Likert parser ────────────────────────────────────────────────────────────

def test_parse_likert_strict_json() -> None:
    raw = '{"faithfulness": 4, "completeness": 5, "coherence": 3}'
    assert _parse_likert_json(raw) == {"faithfulness": 4, "completeness": 5, "coherence": 3}


def test_parse_likert_with_markdown_fences() -> None:
    raw = "```json\n{\"faithfulness\": 2, \"completeness\": 1, \"coherence\": 4}\n```"
    out = _parse_likert_json(raw)
    assert out == {"faithfulness": 2, "completeness": 1, "coherence": 4}


def test_parse_likert_regex_fallback() -> None:
    raw = "faithfulness: 5, completeness: 4, coherence: 5"
    out = _parse_likert_json(raw)
    assert out == {"faithfulness": 5, "completeness": 4, "coherence": 5}


def test_parse_likert_rejects_out_of_range() -> None:
    raw = '{"faithfulness": 7, "completeness": 5, "coherence": 3}'
    assert _parse_likert_json(raw) is None


def test_parse_likert_rejects_missing_field() -> None:
    raw = '{"faithfulness": 4, "completeness": 5}'
    assert _parse_likert_json(raw) is None


# ─── A/B parser ───────────────────────────────────────────────────────────────

@pytest.mark.parametrize("raw,expected", [
    ('{"winner": "A"}',                "A"),
    ('{"winner": "B"}',                "B"),
    ('{"winner": "tie"}',              "tie"),
    ('  ```json\n{"winner": "A"}\n```', "A"),
    ('winner: B',                       "B"),
    ('winner: tie',                     "tie"),
])
def test_parse_ab_valid(raw: str, expected: str) -> None:
    assert _parse_ab_json(raw) == expected


def test_parse_ab_invalid_returns_none() -> None:
    assert _parse_ab_json("I cannot answer.") is None


# ─── Deterministic swap ───────────────────────────────────────────────────────

def test_swap_is_deterministic_in_inputs() -> None:
    a = _deterministic_swap(42, "ctx", "ra", "rb")
    b = _deterministic_swap(42, "ctx", "ra", "rb")
    assert a == b


def test_swap_changes_with_seed() -> None:
    """Different seeds should disagree on at least some inputs."""
    n = 20
    flips = [
        _deterministic_swap(0, "x", f"a{i}", f"b{i}")
        != _deterministic_swap(1, "x", f"a{i}", f"b{i}")
        for i in range(n)
    ]
    assert any(flips), "swap_seed appears to have no effect"


# ─── Score / two-judge aggregation ────────────────────────────────────────────

def test_score_two_judges_aggregation() -> None:
    """
    Two judges each return distinct Likert tuples.  The aggregated mean must
    equal the per-dimension mean and stddev should be non-zero.
    """
    fake = _FakeClient({
        "judge-A": ['{"faithfulness": 5, "completeness": 4, "coherence": 5}'],
        "judge-B": ['{"faithfulness": 3, "completeness": 4, "coherence": 1}'],
    })
    judge = LLMJudge(models=["judge-A", "judge-B"], client=fake)

    out = judge.score(context="ctx", question="q", response="r")

    assert out.faithfulness == pytest.approx(4.0)   # mean(5, 3)
    assert out.completeness == pytest.approx(4.0)   # mean(4, 4)
    assert out.coherence    == pytest.approx(3.0)   # mean(5, 1)
    assert out.judge_std["faithfulness"] > 0
    assert out.judge_std["completeness"] == pytest.approx(0.0)
    assert out.judge_std["coherence"] > 0
    assert set(out.per_judge.keys()) == {"judge-A", "judge-B"}
    assert out.parse_errors == []


def test_score_partial_parse_error_does_not_block_other_judge() -> None:
    """If one judge returns garbage, the other still contributes; the broken judge is logged."""
    fake = _FakeClient({
        "judge-A": ['{"faithfulness": 5, "completeness": 4, "coherence": 5}'],
        "judge-B": ['this is not JSON at all'],
    })
    judge = LLMJudge(models=["judge-A", "judge-B"], client=fake)
    out = judge.score(context="c", question="q", response="r")
    assert out.faithfulness == pytest.approx(5.0)
    assert out.parse_errors == ["judge-B"]


# ─── Pairwise A/B ─────────────────────────────────────────────────────────────

def test_compare_returns_one_result_per_judge() -> None:
    fake = _FakeClient({
        "j1": ['{"winner": "A"}'],
        "j2": ['{"winner": "B"}'],
    })
    judge = LLMJudge(models=["j1", "j2"], client=fake)
    results = judge.compare(context="c", question="q", response_a="A_text", response_b="B_text")
    assert len(results) == 2
    assert {r.judge_model for r in results} == {"j1", "j2"}
    assert all(isinstance(r, PairwiseResult) for r in results)


def test_compare_deswap_invariance() -> None:
    """
    The judge always sees a (potentially) swapped pair, but the returned winner
    must always be in the original A/B frame.  Verify by forcing the fake judge
    to "always pick A" — when the swap flag is True, the recorded winner should
    therefore be 'B' (i.e. the original B was sent first under the relabeling).
    """
    # Pick inputs whose deterministic swap is True so we know the order is flipped.
    a, b = "alpha", "beta"
    swap_flag = _deterministic_swap(42, "ctx", a, b)
    fake = _FakeClient({"j": ['{"winner": "A"}']})
    judge = LLMJudge(models=["j"], client=fake, swap_seed=42)

    [res] = judge.compare(context="ctx", question="q", response_a=a, response_b=b)
    assert res.swap_flag == swap_flag
    if swap_flag:
        # Judge picked A in the swapped frame → original B wins.
        assert res.winner == "B"
    else:
        assert res.winner == "A"
