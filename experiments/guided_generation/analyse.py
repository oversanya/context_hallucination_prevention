"""
Post-experiment analysis for guided beam search.

Reads the artefacts produced by ``run.py`` (outputs.parquet, judge_scores.csv,
pairwise_ab.csv, factscore_turbo.csv) and produces:

* analysis.json     — headline numbers, paired bootstrap CIs, Wilcoxon p-values,
                      length-collapse assertion (audit C4), judge κ (audit C1),
                      latency aggregates.
* effect_sizes.png  — bar chart of mean differences vs vanilla_loop with CIs.
* pareto.png        — tokens/s vs mean faithfulness per condition (audit R8).
* latency_box.png   — wall-time distribution per condition.
* per_task_breakdown.csv — faithfulness mean per (condition, task_type).

Designed to run quickly even when the experiment is partial (e.g. only some
conditions have completed).
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

matplotlib.use("Agg")
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_RESULTS = _PROJECT_ROOT / "experiments" / "guided_generation" / "results"

REFERENCE_CONDITION = "vanilla_loop"
DIMENSIONS = ("faithfulness", "completeness", "coherence")


# ─── Paired bootstrap ─────────────────────────────────────────────────────────

def paired_bootstrap_ci(
    a: np.ndarray, b: np.ndarray, *,
    B: int = 10_000, alpha: float = 0.05, seed: int = 42,
) -> tuple[float, float, float]:
    """
    Compute a paired bootstrap CI for ``mean(b) - mean(a)``.

    Returns (point_estimate, lo, hi) at confidence ``1 - alpha``.
    """
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    mask = np.isfinite(a) & np.isfinite(b)
    a, b = a[mask], b[mask]
    if len(a) < 2:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    diffs = b - a
    point = float(diffs.mean())
    samples = np.empty(B, dtype=float)
    n = len(diffs)
    for k in range(B):
        idx = rng.integers(0, n, n)
        samples[k] = diffs[idx].mean()
    lo, hi = np.quantile(samples, [alpha / 2, 1 - alpha / 2])
    return point, float(lo), float(hi)


def wilcoxon_p(a: np.ndarray, b: np.ndarray) -> float:
    """Two-sided Wilcoxon signed-rank p-value, NaN-safe."""
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    mask = np.isfinite(a) & np.isfinite(b)
    a, b = a[mask], b[mask]
    if len(a) < 5 or np.all(a == b):
        return float("nan")
    try:
        return float(stats.wilcoxon(a, b, zero_method="zsplit", alternative="two-sided").pvalue)
    except ValueError:
        return float("nan")


# ─── Aggregations ─────────────────────────────────────────────────────────────

def per_condition_means(judge: pd.DataFrame) -> pd.DataFrame:
    """Mean Likert score per condition across all rows."""
    out = (judge.groupby("condition")[list(DIMENSIONS)]
           .mean()
           .reset_index())
    return out


def latency_table(outputs: pd.DataFrame) -> pd.DataFrame:
    """Wall-time + token-rate aggregates per condition."""
    df = outputs.copy()
    df["tokens_per_s"] = df["n_response_tokens"] / df["elapsed_s"].replace(0.0, np.nan)
    agg = (df.groupby("condition")
             .agg(mean_elapsed=("elapsed_s", "mean"),
                  median_elapsed=("elapsed_s", "median"),
                  mean_tokens=("n_response_tokens", "mean"),
                  median_tokens=("n_response_tokens", "median"),
                  mean_tokens_per_s=("tokens_per_s", "mean"),
                  n=("response", "count"))
             .reset_index())
    return agg


def length_collapse_check(outputs: pd.DataFrame) -> dict:
    """Audit C4: any condition with median tokens < 0.7× vanilla_loop's median?"""
    medians = outputs.groupby("condition")["n_response_tokens"].median()
    if REFERENCE_CONDITION not in medians.index:
        return {"reference_condition_missing": True}
    ref = float(medians.loc[REFERENCE_CONDITION])
    if ref == 0:
        return {"reference_zero_length": True}
    flagged = {c: float(m) for c, m in medians.items()
               if c != REFERENCE_CONDITION and m < 0.7 * ref}
    return {
        "reference_median_tokens": ref,
        "threshold":               0.7 * ref,
        "flagged_conditions":      flagged,
        "ok":                      len(flagged) == 0,
    }


def pairwise_summary(pairwise: pd.DataFrame) -> pd.DataFrame:
    """Win/loss/tie breakdown per (condition_b, judge_model)."""
    if pairwise.empty:
        return pd.DataFrame()
    grp = pairwise.groupby(["condition_b", "judge_model", "winner"]).size().reset_index(name="count")
    return grp.pivot_table(
        index=["condition_b", "judge_model"], columns="winner", values="count", fill_value=0,
    ).reset_index()


def pairwise_judge_kappa(pairwise: pd.DataFrame) -> dict:
    """Cohen's κ between every pair of judges on the {A, B, tie} winner column."""
    from sklearn.metrics import cohen_kappa_score
    out = {}
    judges = sorted(pairwise["judge_model"].unique())
    for i, j1 in enumerate(judges):
        for j2 in judges[i + 1:]:
            wide = (pairwise[pairwise["judge_model"].isin([j1, j2])]
                    .pivot_table(index=["row_id", "condition_a", "condition_b"],
                                 columns="judge_model", values="winner",
                                 aggfunc="first"))
            wide = wide.dropna()
            if wide.empty:
                out[f"{j1}_vs_{j2}"] = float("nan")
                continue
            try:
                k = float(cohen_kappa_score(wide[j1], wide[j2]))
            except Exception:
                k = float("nan")
            out[f"{j1}_vs_{j2}"] = k
    return out


def likert_judge_kappa(judge_df: pd.DataFrame) -> dict:
    """Per-dimension Cohen κ between the two judges on integer Likert ratings."""
    from sklearn.metrics import cohen_kappa_score
    judges = [c[:-len("_faithfulness")] for c in judge_df.columns
              if c.endswith("_faithfulness")]
    out = {}
    for i, j1 in enumerate(judges):
        for j2 in judges[i + 1:]:
            for dim in DIMENSIONS:
                ca = f"{j1}_{dim}"
                cb = f"{j2}_{dim}"
                if ca not in judge_df.columns or cb not in judge_df.columns:
                    continue
                pair = judge_df[[ca, cb]].dropna()
                try:
                    k = float(cohen_kappa_score(pair[ca].astype(int), pair[cb].astype(int)))
                except Exception:
                    k = float("nan")
                out[f"{j1}_vs_{j2}__{dim}"] = k
    return out


# ─── Plots ────────────────────────────────────────────────────────────────────

def plot_effect_sizes(diffs_table: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    pivot = diffs_table.pivot(index="condition", columns="dimension", values="point")
    err_lo = diffs_table.pivot(index="condition", columns="dimension", values="ci_lo")
    err_hi = diffs_table.pivot(index="condition", columns="dimension", values="ci_hi")

    conditions = pivot.index.tolist()
    x = np.arange(len(conditions))
    width = 0.25
    for i, dim in enumerate(DIMENSIONS):
        if dim not in pivot.columns:
            continue
        means = pivot[dim].values
        lo    = pivot[dim].values - err_lo[dim].values
        hi    = err_hi[dim].values - pivot[dim].values
        ax.bar(x + (i - 1) * width, means, width=width, label=dim, alpha=0.85)
        ax.errorbar(x + (i - 1) * width, means, yerr=[lo, hi], fmt="none", ecolor="black", capsize=3)
    ax.axhline(0, color="black", lw=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(conditions)
    ax.set_ylabel(f"Δ mean Likert vs {REFERENCE_CONDITION}")
    ax.set_title("Effect sizes (95% paired bootstrap CIs)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_pareto(latency: pd.DataFrame, judge_means: pd.DataFrame, out_path: Path) -> None:
    """Tokens/s vs mean faithfulness (audit R8)."""
    merged = latency.merge(judge_means[["condition", "faithfulness"]], on="condition", how="left")
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(merged["mean_tokens_per_s"], merged["faithfulness"], s=80)
    for _, r in merged.iterrows():
        ax.annotate(r["condition"], (r["mean_tokens_per_s"], r["faithfulness"]),
                    textcoords="offset points", xytext=(5, 5), fontsize=9)
    ax.set_xlabel("Mean tokens / second")
    ax.set_ylabel("Mean judge faithfulness")
    ax.set_title("Cost-vs-quality Pareto curve")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_latency_box(outputs: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    order = list(outputs.groupby("condition")["elapsed_s"].median().sort_values().index)
    data = [outputs[outputs["condition"] == c]["elapsed_s"] for c in order]
    ax.boxplot(data, tick_labels=order, showfliers=False)
    ax.set_ylabel("Wall-time per response (s)")
    ax.set_title("Latency distribution by condition")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main(results_dir: Path = _RESULTS) -> None:
    logger.info("Reading results from %s", results_dir)
    outputs_path  = results_dir / "outputs.parquet"
    judge_path    = results_dir / "judge_scores.csv"
    pairwise_path = results_dir / "pairwise_ab.csv"
    fs_path       = results_dir / "factscore_turbo.csv"

    if not outputs_path.exists() or not judge_path.exists():
        logger.error("Missing outputs.parquet or judge_scores.csv at %s", results_dir)
        sys.exit(1)

    outputs   = pd.read_parquet(outputs_path)
    judge     = pd.read_csv(judge_path)
    pairwise  = pd.read_csv(pairwise_path) if pairwise_path.exists() else pd.DataFrame()
    factscore = pd.read_csv(fs_path)       if fs_path.exists()       else pd.DataFrame()

    # Pivot: row × condition × dimension.
    diff_records = []
    if REFERENCE_CONDITION not in judge["condition"].unique():
        logger.error("Reference condition %s missing from judge scores.", REFERENCE_CONDITION)
        sys.exit(1)

    pivot = judge.pivot_table(index="row_id", columns="condition",
                              values=list(DIMENSIONS), aggfunc="first")
    ref = pivot.xs(REFERENCE_CONDITION, level=1, axis=1)
    for cond in [c for c in judge["condition"].unique() if c != REFERENCE_CONDITION]:
        cur = pivot.xs(cond, level=1, axis=1)
        for dim in DIMENSIONS:
            if dim not in cur.columns or dim not in ref.columns:
                continue
            point, lo, hi = paired_bootstrap_ci(ref[dim].values, cur[dim].values)
            p = wilcoxon_p(ref[dim].values, cur[dim].values)
            diff_records.append({
                "condition": cond, "dimension": dim,
                "point": point, "ci_lo": lo, "ci_hi": hi, "wilcoxon_p": p,
            })
    diffs_df = pd.DataFrame(diff_records)
    diffs_df.to_csv(results_dir / "effect_sizes.csv", index=False)

    judge_means = per_condition_means(judge)
    latency     = latency_table(outputs)
    coll        = length_collapse_check(outputs)
    pair_sum    = pairwise_summary(pairwise)
    pair_kappa  = pairwise_judge_kappa(pairwise) if not pairwise.empty else {}
    lik_kappa   = likert_judge_kappa(judge)

    factscore_means = (factscore.groupby("condition")["factscore"].mean().to_dict()
                       if not factscore.empty else {})

    per_task = (judge.groupby(["condition", "task_type"])["faithfulness"]
                .mean().reset_index())
    per_task.to_csv(results_dir / "per_task_breakdown.csv", index=False)
    pair_sum.to_csv(results_dir / "pairwise_summary.csv", index=False)

    analysis = {
        "n_rows":                judge["row_id"].nunique(),
        "conditions":            sorted(judge["condition"].unique().tolist()),
        "reference_condition":   REFERENCE_CONDITION,
        "judge_means":           judge_means.set_index("condition").to_dict(orient="index"),
        "diffs_vs_reference":    diffs_df.to_dict(orient="records"),
        "latency":               latency.to_dict(orient="records"),
        "length_collapse_check": coll,
        "judge_kappa_likert":    lik_kappa,
        "judge_kappa_pairwise":  pair_kappa,
        "factscore_consistency": factscore_means,
        "notes": [
            "factscore_consistency is a SANITY CHECK — gameable by +FS arm (audit R1).",
            "Wilcoxon p-value reported per dim per non-reference condition (alpha=0.05).",
            "Length-collapse flagged when median tokens < 0.7× vanilla_loop median (audit C4).",
        ],
    }
    with open(results_dir / "analysis.json", "w", encoding="utf-8") as fh:
        json.dump(analysis, fh, indent=2, default=float)

    if not diffs_df.empty:
        plot_effect_sizes(diffs_df, results_dir / "effect_sizes.png")
    plot_pareto(latency, judge_means, results_dir / "pareto.png")
    plot_latency_box(outputs, results_dir / "latency_box.png")

    logger.info("Analysis written to %s", results_dir / "analysis.json")
    if coll.get("flagged_conditions"):
        logger.warning("LENGTH COLLAPSE FLAGGED: %s", coll["flagged_conditions"])


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", type=Path, default=_RESULTS)
    args = p.parse_args()
    main(args.results_dir)
