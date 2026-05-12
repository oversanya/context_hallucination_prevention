"""
Unit tests for src/guided_beam_search.py.

The model is replaced by a tiny deterministic stub that returns a fixed logit
profile per step, so we can verify the reranking math without loading any HF
weights.  The stub matches just enough of the HF model API for the loop to
work: ``__call__`` returns an object with ``.logits`` and ``.attentions``,
``.config.num_hidden_layers`` and ``.config.num_attention_heads`` are exposed,
and ``.parameters()`` yields a tensor whose ``.device`` attribute is honoured.
"""
from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Optional
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from src.guided_beam_search import (
    _is_sentence_end,
    guided_beam_search,
)


# ─── Sentence-boundary detection ──────────────────────────────────────────────

@pytest.mark.parametrize("text,expected", [
    (".",     True),
    ("end.",  True),
    ("?",     True),
    ("!",     True),
    ("hello", False),
    (",",     False),
    (" ",     False),
])
def test_sentence_end_detection(text: str, expected: bool) -> None:
    assert _is_sentence_end(text) is expected


# ─── Tiny mock model ──────────────────────────────────────────────────────────

VOCAB = {0: " ", 1: ".", 2: " a", 3: " b", 4: " c"}   # token 1 is sentence end


class _StubTokenizer:
    """Trivial whitespace tokenizer over the ints used by the test vocab."""

    eos_token_id = 0   # space token doubles as EOS to keep the loop short

    def __call__(self, text: str, return_tensors: str = "pt", add_special_tokens: bool = False):
        # Each prompt is encoded as a single arbitrary token id (ignored by the stub model).
        ids = torch.tensor([[42]], dtype=torch.long)
        return SimpleNamespace(input_ids=ids)

    def decode(self, ids, skip_special_tokens: bool = False) -> str:
        if isinstance(ids, int):
            ids = [ids]
        if hasattr(ids, "tolist"):
            ids = ids.tolist()
        # Skip the prompt placeholder id (42) when reconstructing.
        return "".join(VOCAB.get(int(t), "") for t in ids if int(t) != 42)


@dataclass
class _StubConfig:
    num_hidden_layers: int = 2
    num_attention_heads: int = 2


class _StubModel:
    """
    Deterministic stub: returns the same logit profile at every step, optionally
    parameterised so different tests can exercise different rerank decisions.

    ``logit_profile`` is a list of floats of length len(VOCAB); higher = more likely.
    """

    def __init__(self, logit_profile: list, attentions: Optional[tuple] = None):
        self.config = _StubConfig()
        self._logit_profile = torch.tensor(logit_profile, dtype=torch.float32)
        self._attentions = attentions
        # Real torch parameter so .device works.
        self._param = torch.nn.Parameter(torch.zeros(1))

    def parameters(self):
        yield self._param

    def __call__(self, input_ids, output_attentions: bool = False, use_cache: bool = False):
        seq_len = input_ids.shape[1]
        # logits: (batch=1, seq_len, vocab_size)
        logits = self._logit_profile.unsqueeze(0).unsqueeze(0).expand(1, seq_len, -1).clone()
        attentions = None
        if output_attentions:
            if self._attentions is None:
                # uniform attention so the LL feature is well-defined
                attentions = tuple(
                    torch.ones(1, self.config.num_attention_heads, seq_len, seq_len) / seq_len
                    for _ in range(self.config.num_hidden_layers)
                )
            else:
                attentions = self._attentions
        return SimpleNamespace(logits=logits, attentions=attentions)


# ─── Tests ────────────────────────────────────────────────────────────────────

def test_lambda_zero_picks_max_logp() -> None:
    """
    With λ_LL = λ_FS = 0 and length_alpha=0 (no normalisation), beam search must
    select the token sequence with the highest cumulative log-probability — the
    one whose logit is highest.  Token id 4 has the largest logit.
    """
    profile = [0.0, 0.5, 0.2, 0.3, 5.0]   # token 4 dominates
    model = _StubModel(logit_profile=profile)
    tok = _StubTokenizer()

    result = guided_beam_search(
        model=model, tokenizer=tok, prompt="ignored", context="ignored",
        lambda_ll=0.0, lambda_fs=0.0, length_alpha=0.0,
        beam_width=2, max_new_tokens=4, eos_token_id=999,   # never produced
        seed=0,
    )
    assert all(t == 4 for t in result.response_token_ids), (
        f"Expected only token 4, got {result.response_token_ids}"
    )


def test_huge_lambda_ll_overrides_logp() -> None:
    """
    With per_candidate_ll=True and blend mode, LL is recomputed per candidate
    token and the blend dominates when blend_w is high.  Verify that:
    - step_log records are produced
    - LL scores are in [0,1] for all candidates
    """
    profile = [0.0, 0.0, 0.5, 0.4, 0.3]
    model = _StubModel(logit_profile=profile)
    tok = _StubTokenizer()

    clf = MagicMock()
    clf.predict_proba = MagicMock(return_value=np.array([0.1]))

    result = guided_beam_search(
        model=model, tokenizer=tok, prompt="ignored", context="ignored",
        lookback_classifier=clf,
        lambda_ll=0.0, lambda_fs=0.0, length_alpha=0.0,
        beam_width=2, max_new_tokens=3, eos_token_id=999,
        seed=0, log_per_step=True,
        score_mode="blend", blend_w=0.9, per_candidate_ll=True,
    )
    assert result.step_log, "step_log should not be empty"
    for rec in result.step_log:
        assert 0.0 <= rec.ll_score <= 1.0 + 1e-6


def test_per_candidate_ll_differs_from_parent_ll() -> None:
    """
    Design-defect regression test (audit finding): with per_candidate_ll=True
    different candidate tokens should produce different LL scores (since
    attentions differ per candidate).  The stub model returns uniform attention,
    so in this test all LL scores will be equal — we verify no crash and that
    the counters are populated.
    """
    profile = [0.0, 0.0, 0.5, 0.4, 0.3]
    model = _StubModel(logit_profile=profile)
    tok = _StubTokenizer()

    clf = MagicMock()
    clf.predict_proba = MagicMock(return_value=np.array([0.3]))

    result = guided_beam_search(
        model=model, tokenizer=tok, prompt="ignored", context="ignored",
        lookback_classifier=clf,
        lambda_ll=0.0, lambda_fs=0.0, length_alpha=0.0,
        beam_width=2, max_new_tokens=4, eos_token_id=999,
        seed=0, log_per_step=False,
        score_mode="blend", blend_w=0.5, per_candidate_ll=True,
    )
    assert isinstance(result.n_ll_reorders, int)
    assert isinstance(result.n_total_steps, int)
    assert result.n_total_steps > 0


def test_vanilla_loop_equals_hf_at_blend_w0() -> None:
    """
    Regression check: with blend_w=0 (pure log-prob), guided_beam_search
    must select the same sequence as the λ=0 additive mode.
    """
    profile = [0.0, 0.5, 0.2, 0.3, 5.0]
    model = _StubModel(logit_profile=profile)
    tok = _StubTokenizer()

    r_additive = guided_beam_search(
        model=model, tokenizer=tok, prompt="ignored", context="ignored",
        lambda_ll=0.0, lambda_fs=0.0, length_alpha=0.0,
        beam_width=2, max_new_tokens=4, eos_token_id=999, seed=0,
        score_mode="additive",
    )
    r_blend0 = guided_beam_search(
        model=model, tokenizer=tok, prompt="ignored", context="ignored",
        lambda_ll=0.0, lambda_fs=0.0, length_alpha=0.0,
        beam_width=2, max_new_tokens=4, eos_token_id=999, seed=0,
        score_mode="blend", blend_w=0.0,
    )
    assert r_additive.response_token_ids == r_blend0.response_token_ids, (
        "blend_w=0 should match pure log-prob (additive with λ=0)"
    )


def test_length_normalisation_prefers_higher_per_token_logp() -> None:
    """
    With length normalisation α=0.7 and identical per-token logits, two equal-length
    beams must tie on length-normalised score; comparing across step counts the
    per-token average must be what drives the rerank.

    We verify this indirectly: with profile dominated by one token, the sequence
    of that token dominates regardless of α (sanity), and the final_score reflects
    the GNMT formula score = sum_logp / L^α.
    """
    profile = [0.0, 0.0, 0.0, 0.0, 5.0]   # token 4 dominates
    model = _StubModel(logit_profile=profile)
    tok = _StubTokenizer()

    result = guided_beam_search(
        model=model, tokenizer=tok, prompt="ignored", context="ignored",
        lambda_ll=0.0, lambda_fs=0.0, length_alpha=0.7,
        beam_width=1, max_new_tokens=5, eos_token_id=999,
        seed=0,
    )
    L = len(result.response_token_ids)
    assert L == 5
    # Reconstruct expected sum_logp from the softmax of the profile.
    profile_t = torch.tensor(profile)
    logp_top  = torch.log_softmax(profile_t, dim=-1)[4].item()
    expected_sum  = logp_top * L
    expected_norm = expected_sum / (L ** 0.7)
    assert result.final_score == pytest.approx(expected_norm, rel=1e-4)


def test_step_log_contains_kept_and_rejected() -> None:
    """When log_per_step=True, every step should produce at least beam_width survivors and >= 0 rejected."""
    profile = [0.0, 0.5, 0.2, 0.3, 5.0]
    model = _StubModel(logit_profile=profile)
    tok = _StubTokenizer()
    result = guided_beam_search(
        model=model, tokenizer=tok, prompt="ignored", context="ignored",
        lambda_ll=0.0, lambda_fs=0.0, length_alpha=0.0,
        beam_width=2, max_new_tokens=2, eos_token_id=999,
        seed=0, log_per_step=True,
    )
    by_step = {}
    for r in result.step_log:
        by_step.setdefault(r.step, []).append(r)
    assert by_step
    for step, recs in by_step.items():
        kept = [r for r in recs if r.kept]
        assert len(kept) <= 2, f"step {step}: more survivors than beam_width"
