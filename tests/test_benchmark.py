"""
Unit tests for benchmark utilities.

Dataset loading is mocked — no network access required.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.benchmark import compute_metrics, load_ragtruth, load_ragtruth_split


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

def _make_fake_ragtruth_dataset(n: int = 10) -> MagicMock:
    """Construct a minimal fake HuggingFace dataset mimicking RAGTruth schema."""
    import pandas as pd
    from datasets import Dataset

    n_each = n // 2
    fake_df = pd.DataFrame({
        "source_info": ["Context about Paris." * 3] * n,
        "question":    ["Where is Paris?"] * n,
        "response":    [f"Paris response number {i}." for i in range(n)],
        "hallucination_labels_processed": (
            [[] for _ in range(n_each)]
            + [[{"span": f"hallucinated text {i}"}] for i in range(n_each)]
        ),
        "task_type":   ["QA"] * n,
        "model":       ["gpt-4"] * n,
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


# ─── Tests: load_ragtruth_split ───────────────────────────────────────────────

def test_dev_test_split_disjoint() -> None:
    """
    PLAN audit C2: dev and test must be drawn from a single balanced sample
    and must be row-disjoint by construction.  Verifies (a) the split sizes,
    (b) that no response string appears in both subsets, and (c) reproducibility
    under the same seed.
    """
    n_dev, n_test = 4, 6   # small enough that the fake dataset (n=20) fits comfortably
    with patch("src.benchmark.load_dataset", return_value=_make_fake_ragtruth_dataset(n=20)):
        dev1, test1 = load_ragtruth_split(n_dev=n_dev, n_test=n_test, seed=7)
        dev2, test2 = load_ragtruth_split(n_dev=n_dev, n_test=n_test, seed=7)

    assert len(dev1)  == n_dev
    assert len(test1) == n_test

    # Row-disjointness: response strings are unique in the fake dataset, so we
    # can use them as identity tokens.
    assert set(dev1["response"]).isdisjoint(set(test1["response"]))

    # Reproducibility: same seed → same rows in same order.
    pd.testing.assert_frame_equal(dev1.reset_index(drop=True),  dev2.reset_index(drop=True))
    pd.testing.assert_frame_equal(test1.reset_index(drop=True), test2.reset_index(drop=True))
