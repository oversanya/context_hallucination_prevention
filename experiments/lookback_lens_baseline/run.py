"""
Lookback Lens baseline experiment.

Pipeline:
1. Load RAGTruth (n=200, balanced, seed=42).
2. Extract lookback ratio feature vectors via LookbackRatioExtractor.
3. Train LookbackLensClassifier on 80% of samples.
4. Evaluate on held-out 20% and save metrics + classifier.
5. Generate diagnostic plots.

All intermediate feature batches are checkpointed to disk every 50 samples.
"""
from __future__ import annotations

import json
import logging
import random
import sys
import time
from pathlib import Path

import hydra
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from omegaconf import DictConfig
from sklearn.metrics import RocCurveDisplay, roc_curve
from sklearn.model_selection import train_test_split

# Resolve project root so local src/ is importable regardless of cwd.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src import load_ragtruth
from src.lookback_lens import LookbackLensClassifier, LookbackRatioExtractor

matplotlib.use("Agg")
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

_BAR_FMT = "{l_bar}{bar}| {n}/{total} [{elapsed}<{remaining}, {rate_fmt}]"
_CHECKPOINT_INTERVAL = 50  # samples between feature array saves


# ─── Feature extraction with checkpointing ────────────────────────────────────

def _extract_with_checkpoints(
    extractor: LookbackRatioExtractor,
    contexts: list[str],
    responses: list[str],
    out_dir: Path,
    batch_size: int = 8,
) -> np.ndarray:
    """
    Extract features sample-by-sample, saving a .npy checkpoint every
    _CHECKPOINT_INTERVAL samples so the run is resumable.
    """
    n = len(contexts)
    feature_dim = extractor.n_layers * extractor.n_heads
    checkpoint_path = out_dir / "features_checkpoint.npy"

    # Resume from checkpoint if present.
    start_idx = 0
    if checkpoint_path.exists():
        X_saved = np.load(checkpoint_path)
        start_idx = X_saved.shape[0]
        logger.info("Resuming from checkpoint at sample %d/%d", start_idx, n)
        X = np.zeros((n, feature_dim), dtype=np.float32)
        X[:start_idx] = X_saved
    else:
        X = np.zeros((n, feature_dim), dtype=np.float32)

    if start_idx >= n:
        logger.info("All %d samples already extracted — using checkpoint.", n)
        return X

    t0 = time.time()
    for i in range(start_idx, n):
        X[i] = extractor.extract(contexts[i], responses[i])

        # Periodic log.
        if (i + 1) % 25 == 0:
            elapsed = time.time() - t0
            rate = (i - start_idx + 1) / elapsed if elapsed > 0 else float("inf")
            eta = (n - i - 1) / rate if rate > 0 else float("inf")
            logger.info(
                "Progress %d/%d — elapsed %.1fs — ETA %.1fs",
                i + 1, n, elapsed, eta,
            )

        # Checkpoint save.
        if (i + 1) % _CHECKPOINT_INTERVAL == 0 or (i + 1) == n:
            np.save(checkpoint_path, X[: i + 1])
            logger.info("Checkpoint saved at sample %d — %s", i + 1, checkpoint_path)

    return X


# ─── Plotting helpers ─────────────────────────────────────────────────────────

def _plot_ratio_distribution(
    X: np.ndarray,
    y: np.ndarray,
    out_dir: Path,
) -> None:
    """Histogram of mean lookback ratio per sample, separated by ground-truth label."""
    mean_ratios = X.mean(axis=1)  # (n_samples,)
    df_plot = pd.DataFrame({"mean_lookback_ratio": mean_ratios, "label": y})

    fig, ax = plt.subplots(figsize=(7, 4))
    for lbl, grp in df_plot.groupby("label"):
        ax.hist(
            grp["mean_lookback_ratio"],
            bins=25,
            alpha=0.6,
            density=True,
            label="Hallucinated" if lbl == 1 else "Faithful",
        )
    ax.set_xlabel("Mean Lookback Ratio")
    ax.set_ylabel("Density")
    ax.set_title("Lookback Ratio Distribution by Ground-Truth Label")
    ax.legend()
    fig.tight_layout()
    out_path = out_dir / "ratio_distribution.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("Saved %s", out_path)


def _plot_roc_curve(
    y_true: np.ndarray,
    y_score: np.ndarray,
    auc_val: float,
    out_dir: Path,
) -> None:
    """ROC curve for the test split."""
    fpr, tpr, _ = roc_curve(y_true, y_score)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(fpr, tpr, lw=2, label=f"Lookback Lens (AUC = {auc_val:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve — Hallucination Detection (Lookback Lens)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    out_path = out_dir / "roc_curve.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("Saved %s", out_path)


def _plot_layer_heatmap(
    X: np.ndarray,
    n_layers: int,
    n_heads: int,
    out_dir: Path,
) -> None:
    """
    Heatmap of mean lookback ratio per (layer, head) averaged over all samples.
    Rows = layers, columns = heads.
    """
    mean_per_feature = X.mean(axis=0)  # (n_layers * n_heads,)
    matrix = mean_per_feature.reshape(n_layers, n_heads)

    fig, ax = plt.subplots(figsize=(max(6, n_heads), max(4, n_layers // 2)))
    sns.heatmap(
        matrix,
        ax=ax,
        cmap="viridis",
        annot=(n_heads <= 16 and n_layers <= 16),
        fmt=".2f",
        xticklabels=[f"H{h}" for h in range(n_heads)],
        yticklabels=[f"L{l}" for l in range(n_layers)],
    )
    ax.set_xlabel("Attention Head")
    ax.set_ylabel("Layer")
    ax.set_title("Mean Lookback Ratio per Layer and Head")
    fig.tight_layout()
    out_path = out_dir / "layer_head_heatmap.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("Saved %s", out_path)


# ─── Main entry point ─────────────────────────────────────────────────────────

@hydra.main(config_path="../../conf", config_name="lookback_lens", version_base=None)
def main(cfg: DictConfig) -> None:
    # Reproducibility.
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    logger.info("Config: %s", cfg)

    out_dir = _PROJECT_ROOT / cfg.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Dataset ───────────────────────────────────────────────────────────────
    n_total = cfg.n_train + cfg.n_test  # default 200
    df = load_ragtruth(n_samples=n_total, seed=cfg.seed)
    logger.info("Dataset: %d rows loaded", len(df))

    contexts  = df["context"].tolist()
    responses = df["response"].tolist()
    y = df["is_hallucinated"].astype(int).values

    # ── Feature extraction ────────────────────────────────────────────────────
    extractor = LookbackRatioExtractor(
        model_name=cfg.model_name,
        device=cfg.device,
        max_total_tokens=cfg.max_total_tokens,
    )

    X = _extract_with_checkpoints(
        extractor=extractor,
        contexts=contexts,
        responses=responses,
        out_dir=out_dir,
        batch_size=cfg.batch_size,
    )

    logger.info("Feature matrix shape: %s", X.shape)
    np.save(out_dir / "features_full.npy", X)
    np.save(out_dir / "labels.npy", y)

    # ── Train / test split ────────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=cfg.n_test / n_total,
        stratify=y,
        random_state=cfg.seed,
    )
    logger.info(
        "Split: train=%d, test=%d (pos ratio train=%.2f, test=%.2f)",
        len(y_train), len(y_test),
        y_train.mean(), y_test.mean(),
    )

    # ── Train classifier ──────────────────────────────────────────────────────
    clf = LookbackLensClassifier()
    clf.fit(X_train, y_train)

    # ── Evaluate ──────────────────────────────────────────────────────────────
    metrics = clf.score(X_test, y_test)
    metrics["n_train"] = int(len(y_train))
    metrics["n_test"]  = int(len(y_test))
    metrics["model"]   = cfg.model_name

    logger.info("Metrics: %s", json.dumps(metrics, indent=2))

    metrics_path = out_dir / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)
    logger.info("Metrics saved → %s", metrics_path)

    # Save classifier.
    clf_path = out_dir / "classifier.pkl"
    clf.save(clf_path)

    # ── Plots ─────────────────────────────────────────────────────────────────
    proba_test = clf.predict_proba(X_test)

    _plot_ratio_distribution(X, y, out_dir)
    _plot_roc_curve(y_test, proba_test, metrics["roc_auc"], out_dir)
    _plot_layer_heatmap(X, extractor.n_layers, extractor.n_heads, out_dir)

    logger.info("Experiment complete. Results at: %s", out_dir)


if __name__ == "__main__":
    main()
