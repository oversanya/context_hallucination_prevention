"""
Lookback Lens classifier retrained on Qwen-7B-Instruct attention features.

Mirrors experiments/lookback_lens_baseline/run.py but uses the model that will
also be the *generator* for the guided beam search milestone.  This avoids
the cross-model architectural mismatch flagged in PLAN.md §B.

Audit-driven changes (audit_pre.md, 2026-04-29):
* C3 — switch to LogisticRegressionCV with L1 (handled inside the classifier
  class via cfg.classifier.use_cv / penalty / Cs / cv).
* Acceptance: test AUC >= 0.62 AND (train_auc - test_auc) < 0.10.
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
import seaborn as sns
from omegaconf import DictConfig, OmegaConf
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src import load_ragtruth
from src.lookback_lens import LookbackLensClassifier, LookbackRatioExtractor

matplotlib.use("Agg")
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

_CHECKPOINT_INTERVAL = 25     # samples between feature checkpoints
_LOG_EVERY = 10               # samples between progress logs


def _extract_with_checkpoints(
    extractor: LookbackRatioExtractor,
    contexts: list[str],
    responses: list[str],
    out_dir: Path,
) -> np.ndarray:
    """Sample-by-sample extraction with on-disk resumability."""
    n = len(contexts)
    feature_dim = extractor.n_layers * extractor.n_heads
    checkpoint_path = out_dir / "features_checkpoint.npy"

    start_idx = 0
    if checkpoint_path.exists():
        X_saved = np.load(checkpoint_path)
        start_idx = int(X_saved.shape[0])
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

        if (i + 1) % _LOG_EVERY == 0:
            elapsed = time.time() - t0
            rate = (i - start_idx + 1) / elapsed if elapsed > 0 else float("inf")
            eta  = (n - i - 1) / rate if rate > 0 else float("inf")
            logger.info("Progress %d/%d — elapsed %.1fs — ETA %.1fs", i + 1, n, elapsed, eta)

        if (i + 1) % _CHECKPOINT_INTERVAL == 0 or (i + 1) == n:
            np.save(checkpoint_path, X[: i + 1])
            logger.info("Checkpoint saved at sample %d", i + 1)

    return X


def _plot_roc(y_true: np.ndarray, y_score: np.ndarray, auc_val: float, out_dir: Path) -> None:
    fpr, tpr, _ = roc_curve(y_true, y_score)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(fpr, tpr, lw=2, label=f"Lookback Lens — Qwen (AUC = {auc_val:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC — Lookback Lens (Qwen2.5-7B features)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out_dir / "roc_curve.png", dpi=150)
    plt.close(fig)


def _plot_coef_sparsity(clf: LookbackLensClassifier, n_layers: int, n_heads: int, out_dir: Path) -> None:
    """Heatmap of |coefficient| per (layer, head)."""
    coef = np.abs(clf._clf.coef_).reshape(-1)
    matrix = coef.reshape(n_layers, n_heads)
    fig, ax = plt.subplots(figsize=(max(6, n_heads * 0.3), max(4, n_layers * 0.25)))
    sns.heatmap(matrix, ax=ax, cmap="viridis", cbar=True)
    ax.set_xlabel("Attention Head")
    ax.set_ylabel("Layer")
    ax.set_title("Per-(layer,head) classifier weight magnitude (L1)")
    fig.tight_layout()
    fig.savefig(out_dir / "coef_heatmap.png", dpi=150)
    plt.close(fig)


@hydra.main(config_path="../../conf", config_name="lookback_lens_qwen", version_base=None)
def main(cfg: DictConfig) -> None:
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    logger.info("Config:\n%s", OmegaConf.to_yaml(cfg))

    out_dir = _PROJECT_ROOT / cfg.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    n_total = cfg.n_train + cfg.n_test
    df = load_ragtruth(n_samples=n_total, seed=cfg.seed)
    logger.info("Dataset: %d rows", len(df))

    contexts  = df["context"].tolist()
    responses = df["response"].tolist()
    y = np.asarray(df["is_hallucinated"].astype(int).values, dtype=np.int64)

    extractor = LookbackRatioExtractor(
        model_name=cfg.model_name,
        device=cfg.device,
        max_context_tokens=cfg.max_context_tokens,
        max_response_tokens=cfg.max_response_tokens,
        dtype=cfg.dtype,
    )

    X = _extract_with_checkpoints(extractor, contexts, responses, out_dir)
    logger.info("Feature matrix shape: %s", X.shape)
    np.save(out_dir / "features_full.npy", X)
    np.save(out_dir / "labels.npy", y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=cfg.n_test / n_total,
        stratify=y, random_state=cfg.seed,
    )
    logger.info("Split: train=%d, test=%d", len(y_train), len(y_test))

    clf = LookbackLensClassifier(
        max_iter=cfg.classifier.max_iter,
        use_cv=cfg.classifier.use_cv,
        Cs=list(cfg.classifier.Cs),
        penalty=cfg.classifier.penalty,
        cv=cfg.classifier.cv,
    )
    clf.fit(X_train, y_train)

    metrics = clf.score(X_test, y_test)
    train_proba = clf.predict_proba(X_train)
    metrics["train_auc"]      = float(roc_auc_score(y_train, train_proba))
    metrics["test_auc"]       = metrics["roc_auc"]
    metrics["train_test_gap"] = metrics["train_auc"] - metrics["test_auc"]
    metrics["n_train"]        = int(len(y_train))
    metrics["n_test"]         = int(len(y_test))
    metrics["n_features"]     = int(X.shape[1])
    metrics["model"]          = cfg.model_name
    metrics["chosen_C"]       = clf.chosen_C
    metrics["n_nonzero_coef"] = clf.n_nonzero_coef

    # Acceptance check (audit C3).
    accepted = (metrics["test_auc"] >= cfg.acceptance.min_test_auc
                and metrics["train_test_gap"] < cfg.acceptance.max_train_test_gap)
    metrics["accepted"] = bool(accepted)
    metrics["acceptance_min_test_auc"]      = float(cfg.acceptance.min_test_auc)
    metrics["acceptance_max_train_test_gap"] = float(cfg.acceptance.max_train_test_gap)

    logger.info("Metrics:\n%s", json.dumps(metrics, indent=2))
    if not accepted:
        logger.warning(
            "ACCEPTANCE FAIL: test_auc=%.3f (need >= %.3f), gap=%.3f (need < %.3f)",
            metrics["test_auc"], cfg.acceptance.min_test_auc,
            metrics["train_test_gap"], cfg.acceptance.max_train_test_gap,
        )

    with open(out_dir / "metrics.json", "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)

    clf.save(out_dir / "classifier.pkl")

    test_proba = clf.predict_proba(X_test)
    _plot_roc(y_test, test_proba, metrics["test_auc"], out_dir)
    _plot_coef_sparsity(clf, extractor.n_layers, extractor.n_heads, out_dir)

    try:
        from comet_ml import Experiment as CometExperiment
        comet = CometExperiment(
            api_key="WUHdQW2NyhxNGhwB9goVTy3Hi",
            project_name="context-hallucination-prevention",
            workspace="vekshinkir",
            auto_output_logging="simple",
        )
        comet.set_name("lookback-lens-qwen-retrain")
        comet.log_metrics(metrics)
        comet.log_parameter("model", cfg.model_name)
        comet.log_asset(str(out_dir / "roc_curve.png"), file_name="roc_curve.png")
        comet.log_asset(str(out_dir / "coef_heatmap.png"), file_name="coef_heatmap.png")
        comet.end()
    except Exception as e:
        logger.warning("CometML logging failed (non-fatal): %s", e)

    logger.info("Done — results at %s (accepted=%s)", out_dir, accepted)


if __name__ == "__main__":
    main()
