"""
Unit tests for Lookback Lens.

All model forward passes are monkeypatched — no GPU or HuggingFace download required.
"""
from __future__ import annotations

import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from src.lookback_lens import LookbackLensClassifier, LookbackRatioExtractor


# ─── Helpers ──────────────────────────────────────────────────────────────────

N_LAYERS = 4
N_HEADS = 8
FEATURE_DIM = N_LAYERS * N_HEADS  # 32
SEQ_LEN = 20
N_CTX_TOKENS = 8  # first 8 tokens are context


def _make_fake_attentions(batch=1, n_heads=N_HEADS, seq_len=SEQ_LEN, n_layers=N_LAYERS):
    """Uniform attention weights summing to 1 along the key axis."""
    attn_list = []
    for _ in range(n_layers):
        # (batch, n_heads, seq_len, seq_len) — uniform
        a = torch.ones(batch, n_heads, seq_len, seq_len) / seq_len
        attn_list.append(a)
    return tuple(attn_list)


def _make_extractor_with_mock_model() -> LookbackRatioExtractor:
    """Build a LookbackRatioExtractor whose model/tokenizer are fully mocked."""
    with patch("src.lookback_lens.extractor.AutoTokenizer.from_pretrained") as mock_tok, \
         patch("src.lookback_lens.extractor.AutoModelForCausalLM.from_pretrained") as mock_mdl:

        # --- tokenizer mock ---
        tokenizer = MagicMock()
        tokenizer.pad_token = None
        tokenizer.eos_token = "<eos>"

        def tokenize_side_effect(text, **kwargs):
            # Return a fake sequence of N_CTX_TOKENS token ids for any input.
            ids = list(range(N_CTX_TOKENS))
            if kwargs.get("return_tensors") == "pt":
                # Return mapping with tensors (simulates joint encoding).
                ids_full = list(range(SEQ_LEN))
                mock_enc = MagicMock()
                mock_enc.__getitem__ = lambda self, k: (
                    torch.tensor([ids_full]) if k == "input_ids"
                    else torch.ones(1, SEQ_LEN, dtype=torch.long)
                )
                return mock_enc
            return {"input_ids": ids}

        tokenizer.side_effect = tokenize_side_effect
        mock_tok.return_value = tokenizer

        # --- model mock ---
        model = MagicMock()
        model.config.num_hidden_layers = N_LAYERS
        model.config.num_attention_heads = N_HEADS

        fake_attentions = _make_fake_attentions()

        def model_forward(**kwargs):
            out = MagicMock()
            out.attentions = fake_attentions
            return out

        model.return_value = model_forward()
        model.to.return_value = model
        mock_mdl.return_value = model

        extractor = LookbackRatioExtractor(
            model_name="mock-model",
            device="cpu",
            max_context_chars=500,
            max_total_tokens=SEQ_LEN,
        )

    return extractor


# ─── Tests: Extractor ─────────────────────────────────────────────────────────

def test_extractor_output_shape() -> None:
    """extract() must return a 1-D vector of length n_layers * n_heads."""
    extractor = _make_extractor_with_mock_model()

    # Directly mock the model forward call so we control attentions.
    fake_attentions = _make_fake_attentions()

    mock_output = MagicMock()
    mock_output.attentions = fake_attentions

    with patch.object(extractor.model, "__call__", return_value=mock_output):
        # Re-mock tokenizer calls used inside extract().
        extractor.tokenizer.side_effect = None

        def tok_call(text, **kwargs):
            if kwargs.get("return_tensors") == "pt":
                enc = MagicMock()
                enc.__getitem__ = lambda self, k: (
                    torch.tensor([list(range(SEQ_LEN))]) if k == "input_ids"
                    else torch.ones(1, SEQ_LEN, dtype=torch.long)
                )
                return enc
            return {"input_ids": list(range(N_CTX_TOKENS))}

        extractor.tokenizer.side_effect = tok_call

        features = extractor.extract("some context text", "some response text")

    assert features.shape == (FEATURE_DIM,), (
        f"Expected shape ({FEATURE_DIM},), got {features.shape}"
    )


def test_lookback_ratio_sums_to_one() -> None:
    """
    With uniform attention (each token equally attended), the lookback ratio
    for context tokens equals n_ctx_tokens / seq_len and lies in [0, 1].
    """
    extractor = _make_extractor_with_mock_model()
    fake_attentions = _make_fake_attentions()

    mock_output = MagicMock()
    mock_output.attentions = fake_attentions

    with patch.object(extractor.model, "__call__", return_value=mock_output):
        extractor.tokenizer.side_effect = None

        def tok_call(text, **kwargs):
            if kwargs.get("return_tensors") == "pt":
                enc = MagicMock()
                enc.__getitem__ = lambda self, k: (
                    torch.tensor([list(range(SEQ_LEN))]) if k == "input_ids"
                    else torch.ones(1, SEQ_LEN, dtype=torch.long)
                )
                return enc
            return {"input_ids": list(range(N_CTX_TOKENS))}

        extractor.tokenizer.side_effect = tok_call
        features = extractor.extract("some context text", "some response text")

    # All ratios must lie in [0, 1].
    assert np.all(features >= 0.0), "Negative lookback ratio detected."
    assert np.all(features <= 1.0 + 1e-6), "Lookback ratio > 1 detected."

    # With uniform attention: expected ratio = N_CTX_TOKENS / SEQ_LEN.
    expected = N_CTX_TOKENS / SEQ_LEN
    np.testing.assert_allclose(features, expected, atol=1e-5)


# ─── Tests: Classifier ────────────────────────────────────────────────────────

@pytest.fixture()
def synthetic_data():
    """Simple linearly-separable synthetic dataset (n=100, dim=32)."""
    rng = np.random.default_rng(0)
    n = 100
    X = rng.standard_normal((n, FEATURE_DIM)).astype(np.float32)
    # Positive class (hallucinated) has higher mean on first feature.
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
    proba_orig = clf.predict_proba(X)
    proba_loaded = clf2.predict_proba(X)

    np.testing.assert_array_equal(proba_orig, proba_loaded)
