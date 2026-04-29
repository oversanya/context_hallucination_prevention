"""
Unit tests for Lookback Lens.

All model forward passes are monkeypatched — no GPU or HuggingFace download required.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from src.lookback_lens import LookbackLensClassifier, LookbackRatioExtractor


# ─── Constants ────────────────────────────────────────────────────────────────

N_LAYERS = 4
N_HEADS = 8
FEATURE_DIM = N_LAYERS * N_HEADS   # 32
N_CTX_TOKENS = 8
N_RESP_TOKENS = 12
SEQ_LEN = N_CTX_TOKENS + N_RESP_TOKENS   # 20


def _make_fake_attentions(seq_len: int = SEQ_LEN):
    """Uniform attention weights summing to 1 along the key axis."""
    return tuple(
        torch.ones(1, N_HEADS, seq_len, seq_len) / seq_len
        for _ in range(N_LAYERS)
    )


def _make_extractor_with_mock_model() -> LookbackRatioExtractor:
    """Return a LookbackRatioExtractor with fully mocked model and tokenizer."""
    with patch("src.lookback_lens.extractor.AutoTokenizer.from_pretrained") as mock_tok, \
         patch("src.lookback_lens.extractor.AutoModelForCausalLM.from_pretrained") as mock_mdl:

        tokenizer = MagicMock()
        tokenizer.pad_token = None
        tokenizer.eos_token = "<eos>"
        tokenizer.side_effect = None   # overridden inside each test
        mock_tok.return_value = tokenizer

        model = MagicMock()
        model.config.num_hidden_layers = N_LAYERS
        model.config.num_attention_heads = N_HEADS
        model.to.return_value = model
        mock_mdl.return_value = model

        extractor = LookbackRatioExtractor(
            model_name="mock-model",
            device="cpu",
            max_context_tokens=N_CTX_TOKENS,
            max_response_tokens=N_RESP_TOKENS,
        )

    return extractor


def _two_call_tokenizer(ctx_len: int = N_CTX_TOKENS, resp_len: int = N_RESP_TOKENS):
    """
    Return a tokenizer side_effect that handles the two separate calls in extract():
      call 1 (context)  → dict with pt tensors of shape (1, ctx_len)
      call 2 (response) → dict with pt tensors of shape (1, resp_len)
    Returns plain dicts so dict subscript gives real tensors with .shape.
    """
    call_count = [0]

    def _tok(text, **kwargs):
        call_count[0] += 1
        length = ctx_len if call_count[0] == 1 else resp_len
        return {
            "input_ids":      torch.tensor([list(range(length))]),
            "attention_mask": torch.ones(1, length, dtype=torch.long),
        }

    return _tok


# ─── Tests: Extractor ─────────────────────────────────────────────────────────

def _set_model_return(extractor: LookbackRatioExtractor, attentions: tuple) -> None:
    """Wire fake attention tensors into the model mock's return value."""
    extractor.model.return_value.attentions = attentions


def test_extractor_output_shape() -> None:
    """extract() must return a 1-D vector of length n_layers * n_heads."""
    extractor = _make_extractor_with_mock_model()
    _set_model_return(extractor, _make_fake_attentions())
    extractor.tokenizer.side_effect = _two_call_tokenizer()

    features = extractor.extract("some context text", "some response text")

    assert features.shape == (FEATURE_DIM,), (
        f"Expected ({FEATURE_DIM},), got {features.shape}"
    )


def test_lookback_ratio_in_unit_interval() -> None:
    """
    With uniform attention, lookback ratio = ctx_len / seq_len for every head.
    All ratios must lie in [0, 1].
    """
    extractor = _make_extractor_with_mock_model()
    _set_model_return(extractor, _make_fake_attentions())
    extractor.tokenizer.side_effect = _two_call_tokenizer()

    features = extractor.extract("some context text", "some response text")

    assert np.all(features >= 0.0), "Negative lookback ratio detected."
    assert np.all(features <= 1.0 + 1e-6), "Lookback ratio > 1 detected."
    expected = N_CTX_TOKENS / SEQ_LEN
    np.testing.assert_allclose(features, expected, atol=1e-5)


def test_extractor_no_zero_vectors_on_long_context() -> None:
    """
    PLAN §A1 regression test: separate per-budget tokenization must guarantee
    non-zero features even when context fills its entire token budget.
    """
    extractor = _make_extractor_with_mock_model()
    _set_model_return(extractor, _make_fake_attentions(seq_len=N_CTX_TOKENS + N_RESP_TOKENS))
    extractor.tokenizer.side_effect = _two_call_tokenizer(
        ctx_len=N_CTX_TOKENS,
        resp_len=N_RESP_TOKENS,
    )

    features = extractor.extract("a" * 10_000, "short response")

    assert features.sum() > 0, (
        "Zero feature vector returned despite non-empty context and response — "
        "context-truncation bug not fixed."
    )
    assert features.shape == (FEATURE_DIM,)


# ─── Tests: Classifier ────────────────────────────────────────────────────────

@pytest.fixture()
def synthetic_data():
    """Simple linearly-separable dataset (n=100, dim=32)."""
    rng = np.random.default_rng(0)
    X = rng.standard_normal((100, FEATURE_DIM)).astype(np.float32)
    y = (X[:, 0] > 0).astype(int)
    return X, y


def test_classifier_fit_predict(synthetic_data) -> None:
    """predict_proba must return values in [0, 1] after fitting."""
    X, y = synthetic_data
    clf = LookbackLensClassifier()
    clf.fit(X, y)
    proba = clf.predict_proba(X)
    assert proba.shape == (len(y),)
    assert np.all(proba >= 0.0) and np.all(proba <= 1.0)


def test_classifier_save_load(synthetic_data, tmp_path: Path) -> None:
    """Saved and reloaded classifier must produce identical predictions."""
    X, y = synthetic_data
    clf = LookbackLensClassifier()
    clf.fit(X, y)

    save_path = tmp_path / "clf.pkl"
    clf.save(save_path)

    clf2 = LookbackLensClassifier.load(save_path)
    np.testing.assert_array_equal(clf.predict_proba(X), clf2.predict_proba(X))
