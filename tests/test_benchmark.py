"""
Unit tests for benchmark utilities.

Dataset loading is mocked — no network access required.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.benchmark import compute_metrics, load_ragtruth


# ─── Tests: compute_metrics ───────────────────────────────────────────────────

def test_compute_metrics_basic() -> None:
    """compute_metrics must return a dict containing roc_auc on a synthetic DataFrame."""
    df = pd.DataFrame({
        "factscore":      [0.9, 0.8, 0.3, 0.2, 0.7, 0.1, 0.85, 0.15],
        "is_hallucinated": [False, False, True, True, False, True, False, True],
        "n_facts":         [3,    4,    2,   5,   3,   1,   4,    2],
    })
    metrics = compute_metrics(df, threshold=0.5)

    assert "roc_auc" in metrics
    assert metrics["roc_auc"] is not None
    assert 0.0 <= metrics["roc_auc"] <= 1.0
    assert "f1" in metrics
    assert metrics["n_samples"] == 8


# ─── Tests: load_ragtruth ─────────────────────────────────────────────────────

def _make_fake_ragtruth_dataset() -> MagicMock:
    """Construct a minimal fake HuggingFace dataset mimicking RAGTruth schema."""
    import pandas as pd
    from datasets import Dataset

    fake_df = pd.DataFrame({
        "source_info": ["Context about Paris." * 3] * 10,
        "question":    ["Where is Paris?"] * 10,
        "response":    ["Paris is in France."] * 10,
        "hallucination_labels_processed": [
            [],          # faithful
            [{"span": "hallucinated text"}],  # hallucinated
        ] * 5,
        "task_type":   ["QA"] * 10,
        "model":       ["gpt-4"] * 10,
    })
    fake_hf = Dataset.from_pandas(fake_df)
    mock_ds = MagicMock()
    mock_ds.keys.return_value = ["train"]
    mock_ds.__getitem__ = MagicMock(return_value=fake_hf)
    return mock_ds


def test_load_ragtruth_smoke() -> None:
    """load_ragtruth() must return a DataFrame with the canonical schema columns."""
    expected_columns = {"context", "question", "response", "is_hallucinated", "task_type", "gen_model", "source"}

    with patch("src.benchmark.load_dataset", return_value=_make_fake_ragtruth_dataset()):
        df = load_ragtruth(n_samples=None, seed=42)

    assert isinstance(df, pd.DataFrame)
    assert expected_columns.issubset(set(df.columns)), (
        f"Missing columns: {expected_columns - set(df.columns)}"
    )
    assert df["source"].unique().tolist() == ["ragtruth"]
    assert df["is_hallucinated"].dtype == bool or df["is_hallucinated"].dtype == object
