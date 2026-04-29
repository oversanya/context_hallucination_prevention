"""
FActScore-Turbo baseline experiment.

Loads RAGTruth, scores every sample, computes detection metrics,
saves CSV + JSON results, and generates diagnostic plots.
"""
from __future__ import annotations

import json
import logging
import random
import sys
from pathlib import Path

import hydra
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from omegaconf import DictConfig
from sklearn.metrics import RocCurveDisplay, roc_curve

# Resolve project root so local src/ is importable regardless of cwd.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src import FActScoreTurbo, load_ragtruth, run_factscore_benchmark

matplotlib.use("Agg")
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


# ─── Plotting helpers ─────────────────────────────────────────────────────────

def _plot_factscore_distribution(df: pd.DataFrame, out_dir: Path) -> None:
    """Histogram of FActScore values separated by ground-truth label."""
    fig, ax = plt.subplots(figsize=(7, 4))
    for label, grp in df.groupby("is_hallucinated"):
        ax.hist(
            grp["factscore"].dropna(),
            bins=25,
            alpha=0.6,
            label="Hallucinated" if label else "Faithful",
            density=True,
        )
    ax.axvline(0.5, color="black", linestyle="--", linewidth=1, label="threshold=0.5")
    ax.set_xlabel("FActScore")
    ax.set_ylabel("Density")
    ax.set_title("FActScore Distribution by Ground-Truth Label")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "factscore_distribution.png", dpi=150)
    plt.close(fig)
    logger.info("Saved factscore_distribution.png")


def _plot_roc_curve(df: pd.DataFrame, metrics: dict, out_dir: Path) -> None:
    """ROC curve for hallucination detection (1 - FActScore as score)."""
    valid = df.dropna(subset=["factscore", "is_hallucinated"])
    y_true = valid["is_hallucinated"].astype(int).values
    y_score = (1 - valid["factscore"]).values

    fpr, tpr, _ = roc_curve(y_true, y_score)
    auc_val = metrics.get("roc_auc", float("nan"))

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(fpr, tpr, lw=2, label=f"FActScore-Turbo (AUC = {auc_val:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve — Hallucination Detection")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out_dir / "roc_curve.png", dpi=150)
    plt.close(fig)
    logger.info("Saved roc_curve.png")


def _plot_per_task_f1(df: pd.DataFrame, out_dir: Path, threshold: float = 0.5, label: str = "") -> None:
    """Bar chart of binary F1 per task_type."""
    from sklearn.metrics import f1_score

    records = []
    for task, grp in df.groupby("task_type"):
        valid = grp.dropna(subset=["factscore", "is_hallucinated"])
        if len(valid) < 5:  # too few samples for reliable metric
            continue
        y_true = valid["is_hallucinated"].astype(int).values
        y_pred = (valid["factscore"] < threshold).astype(int).values
        records.append({"task_type": task, "f1": f1_score(y_true, y_pred, zero_division=0)})

    if not records:
        logger.warning("No task groups with >= 5 samples — skipping per-task F1 plot")
        return

    task_df = pd.DataFrame(records).sort_values("f1", ascending=False)

    fig, ax = plt.subplots(figsize=(max(5, len(task_df) * 1.2), 4))
    sns.barplot(data=task_df, x="task_type", y="f1", ax=ax, palette="viridis")
    ax.set_ylim(0, 1)
    ax.set_xlabel("Task Type")
    ax.set_ylabel("Binary F1")
    title_suffix = label if label else f"τ = {threshold:.3f}"
    ax.set_title(f"Per-Task Hallucination Detection F1 ({title_suffix})")
    fig.tight_layout()
    fig.savefig(out_dir / "per_task_f1.png", dpi=150)
    plt.close(fig)
    logger.info("Saved per_task_f1.png")


# ─── Main entry point ─────────────────────────────────────────────────────────

@hydra.main(config_path="../../conf", config_name="factscore_turbo", version_base=None)
def main(cfg: DictConfig) -> None:
    # Reproducibility
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    logger.info("Config: %s", cfg)

    out_dir = _PROJECT_ROOT / cfg.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load dataset ──────────────────────────────────────────────────────────
    if cfg.dataset == "ragtruth":
        df = load_ragtruth(n_samples=cfg.n_samples, seed=cfg.seed)
    else:
        raise ValueError(f"Unknown dataset: {cfg.dataset!r}")

    logger.info("Dataset loaded: %d rows", len(df))

    # ── Instantiate scorer ────────────────────────────────────────────────────
    scorer = FActScoreTurbo(
        model=cfg.scorer.model,
        temperature=cfg.scorer.temperature,
        max_facts=cfg.scorer.max_facts,
        batch_verify=cfg.scorer.batch_verify,
    )

    # ── Run benchmark ─────────────────────────────────────────────────────────
    save_path = out_dir / "results"
    results_df, metrics = run_factscore_benchmark(
        df=df,
        scorer=scorer,
        save_path=save_path,
        seed=cfg.seed,
    )

    logger.info("Metrics: %s", json.dumps(metrics, indent=2))

    # ── Plots ─────────────────────────────────────────────────────────────────
    optimal_tau = metrics.get("optimal_threshold", 0.5)
    _plot_factscore_distribution(results_df, out_dir)
    _plot_roc_curve(results_df, metrics, out_dir)
    _plot_per_task_f1(results_df, out_dir, threshold=optimal_tau, label=f"optimal τ = {optimal_tau:.3f}")

    logger.info("Experiment complete. Results at: %s", out_dir)


if __name__ == "__main__":
    main()
