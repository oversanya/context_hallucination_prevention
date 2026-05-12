"""
Final guided-generation evaluation on the held-out test split (n=200).

Runs five conditions per test row:
    vanilla_hf     — HF model.generate (audit R3 sanity check)
    vanilla_loop   — custom-loop with lambda_ll = lambda_fs = 0
    lookback       — only LL term active
    factscore      — only FS term active
    combined       — both terms active

After all generations are written to outputs.parquet, the script scores every
response with the cross-family judge ensemble, runs three pairwise A/B
comparisons (vanilla_loop vs each of +LL, +FS, +LL+FS), and computes a
FActScore-Turbo consistency check using qwen2.5:7b as a third-party scorer
distinct from both generator and judges.

Per-row checkpointing of outputs.parquet every ``cfg.checkpoint_every`` rows
ensures that a crash never costs more than that many generations.
"""
from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src import FActScoreTurbo, LLMJudge, load_ragtruth_split

from experiments.guided_generation.lib import (
    build_prompt,
    generate_condition,
    load_completed_keys,
    load_generator,
    load_lookback_classifier,
    make_conditions,
    save_outputs_incremental,
    write_splits_manifest,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

_COMET_KEY  = "WUHdQW2NyhxNGhwB9goVTy3Hi"
_COMET_PROJ = "context-hallucination-prevention"
_COMET_WS   = "vekshinkir"


# ─── Stage 1: generation ──────────────────────────────────────────────────────

def stage_generate(cfg: DictConfig, out_dir: Path, comet=None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate all five conditions on the test split with checkpointing."""
    dev, test = load_ragtruth_split(
        n_dev=cfg.dataset.n_dev, n_test=cfg.dataset.n_test, seed=cfg.seed,
        task_filter=cfg.dataset.get("task_filter", None),
    )
    write_splits_manifest(dev, test, out_dir)

    outputs_path = out_dir / "outputs.parquet"
    completed = load_completed_keys(outputs_path)
    logger.info("Resuming — %d (row, condition) pairs already complete.", len(completed))

    # Best lambdas from the sweep, falling back to config defaults if missing.
    best_path = out_dir / "best_lambdas.json"
    if best_path.exists():
        best = json.loads(best_path.read_text())
    else:
        logger.warning("best_lambdas.json missing — using config defaults.")
        best = {"lambda_ll": float(cfg.lookback.lambda_ll),
                "lambda_fs": float(cfg.factscore.lambda_fs)}
    logger.info("Using λ_LL=%.3f, λ_FS=%.3f", best["lambda_ll"], best["lambda_fs"])

    conditions = make_conditions(
        best["lambda_ll"], best["lambda_fs"],
        score_mode=cfg.lookback.get("score_mode", "blend"),
        blend_w=float(cfg.lookback.get("blend_w", 0.5)),
        per_candidate_ll=bool(cfg.lookback.get("per_candidate_ll", True)),
    )
    logger.info("Conditions: %s", [c.name for c in conditions])

    model, tokenizer = load_generator(
        cfg.generator.model, cfg.generator.device, cfg.generator.dtype,
    )
    classifier = load_lookback_classifier(_PROJECT_ROOT / cfg.lookback.classifier_path)
    fs_scorer  = FActScoreTurbo(
        model=cfg.factscore.scorer_model, temperature=0.0,
        max_facts=10, batch_verify=True,
    )

    rows: list[dict] = []
    if outputs_path.exists():
        rows = pd.read_parquet(outputs_path).to_dict(orient="records")

    t_start = time.time()
    new_since_checkpoint = 0
    for ridx, row in test.iterrows():
        prompt = build_prompt(tokenizer, row["context"], row.get("question", ""))
        for cond in conditions:
            if (int(ridx), cond.name) in completed:
                continue
            try:
                out = generate_condition(
                    cond=cond,
                    model=model, tokenizer=tokenizer,
                    classifier=classifier, fs_scorer=fs_scorer,
                    prompt=prompt, context=row["context"],
                    beam_width=cfg.generator.beam_width,
                    max_new_tokens=cfg.generator.max_new_tokens,
                    length_alpha=cfg.generator.length_alpha,
                    seed=cfg.seed,
                    log_per_step=False,    # keep parquet small; §D notebook re-runs with True for 5 examples
                    fs_neutral=cfg.factscore.neutral_score_when_no_facts,
                )
            except Exception as exc:  # noqa: BLE001
                logger.error("row=%d cond=%s failed: %s", ridx, cond.name, exc)
                continue

            row_rec = {
                "row_id":              int(ridx),
                "task_type":           row.get("task_type", "Unknown"),
                "context":             row["context"],
                "question":            row.get("question", "") or "",
                "is_hallucinated_gt":  bool(row.get("is_hallucinated", False)),
                "condition":           cond.name,
                "response":            out["response"],
                "n_response_tokens":   int(out["n_response_tokens"]),
                "elapsed_s":           float(out["elapsed_s"]),
                "lambda_ll":           float(out["lambda_ll"]),
                "lambda_fs":           float(out["lambda_fs"]),
                "final_score":         float(out["final_score"])
                                       if not np.isnan(out["final_score"]) else None,
            }
            rows.append(row_rec)
            new_since_checkpoint += 1

            # Real-time CometML logging per generation
            if comet is not None:
                step_idx = len(rows)
                comet.log_metrics({
                    f"gen/{cond.name}/elapsed_s":       float(out["elapsed_s"]),
                    f"gen/{cond.name}/n_tokens":        int(out["n_response_tokens"]),
                }, step=step_idx)
                comet.log_text(
                    f"[row={ridx} cond={cond.name}] {out['response'][:400]}",
                    metadata={"row_id": int(ridx), "condition": cond.name,
                              "task_type": row.get("task_type", "")},
                )
                logger.info("[CometML] logged row=%d cond=%s", ridx, cond.name)

            if new_since_checkpoint >= cfg.checkpoint_every:
                save_outputs_incremental(rows, outputs_path)
                new_since_checkpoint = 0
                logger.info("[%.0fs] checkpoint @ row=%d cond=%s — total rows=%d",
                            time.time() - t_start, ridx, cond.name, len(rows))

    save_outputs_incremental(rows, outputs_path)
    df_out = pd.DataFrame(rows)
    logger.info("Stage 1 done — %d total response rows across %d conditions.",
                len(df_out), df_out["condition"].nunique())
    return df_out, test


# ─── Stage 2: judging ─────────────────────────────────────────────────────────

def stage_judge(cfg: DictConfig, out_dir: Path, outputs_df: pd.DataFrame, comet=None) -> None:
    """Score every response with both judges; persist judge_scores.csv."""
    judge_path = out_dir / "judge_scores.csv"
    if judge_path.exists():
        existing = pd.read_csv(judge_path)
        scored_keys = set(zip(existing["row_id"].astype(int),
                              existing["condition"].astype(str)))
        records = existing.to_dict(orient="records")
    else:
        scored_keys = set()
        records = []

    judge = LLMJudge(models=list(cfg.judge.models), swap_seed=cfg.judge.swap_seed)

    for _, r in outputs_df.iterrows():
        key = (int(r["row_id"]), str(r["condition"]))
        if key in scored_keys:
            continue
        try:
            out = judge.score(
                context=str(r["context"]),
                question=str(r.get("question", "")),
                response=str(r["response"]),
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("Judge failed on %s: %s", key, exc)
            continue

        rec = {
            "row_id":         int(r["row_id"]),
            "condition":      r["condition"],
            "task_type":      r.get("task_type", "Unknown"),
            "faithfulness":   out.faithfulness,
            "completeness":   out.completeness,
            "coherence":      out.coherence,
            "judge_std_faith": out.judge_std.get("faithfulness", float("nan")),
            "judge_std_comp":  out.judge_std.get("completeness", float("nan")),
            "judge_std_coh":   out.judge_std.get("coherence",    float("nan")),
            "parse_errors":   ",".join(out.parse_errors),
        }
        for m, scores in out.per_judge.items():
            rec[f"{m}_faithfulness"] = scores.get("faithfulness")
            rec[f"{m}_completeness"] = scores.get("completeness")
            rec[f"{m}_coherence"]    = scores.get("coherence")
        records.append(rec)
        if comet is not None:
            step_idx = len(records)
            comet.log_metrics({
                f"judge/{r['condition']}/faithfulness": rec["faithfulness"],
                f"judge/{r['condition']}/completeness": rec["completeness"],
                f"judge/{r['condition']}/coherence":    rec["coherence"],
            }, step=step_idx)
        if len(records) % 20 == 0:
            pd.DataFrame(records).to_csv(judge_path, index=False)

    pd.DataFrame(records).to_csv(judge_path, index=False)
    logger.info("Judge scores saved → %s (n=%d)", judge_path, len(records))


# ─── Stage 3: pairwise A/B ────────────────────────────────────────────────────

def stage_pairwise(cfg: DictConfig, out_dir: Path, outputs_df: pd.DataFrame) -> None:
    """A/B preference: vanilla_loop vs (+LL, +FS, +LL+FS).  Audit R2."""
    pairs = [
        ("vanilla_loop", "lookback"),
        ("vanilla_loop", "factscore"),
        ("vanilla_loop", "combined"),
    ]
    pairwise_path = out_dir / "pairwise_ab.csv"
    if pairwise_path.exists():
        existing = pd.read_csv(pairwise_path)
        done_keys = set(zip(existing["row_id"].astype(int),
                            existing["condition_a"], existing["condition_b"],
                            existing["judge_model"]))
        records = existing.to_dict(orient="records")
    else:
        done_keys = set()
        records = []

    judge = LLMJudge(models=list(cfg.judge.models), swap_seed=cfg.judge.swap_seed)
    by_row = {(int(r["row_id"]), r["condition"]): r for _, r in outputs_df.iterrows()}

    for ridx in sorted({int(r["row_id"]) for _, r in outputs_df.iterrows()}):
        for cond_a, cond_b in pairs:
            if (ridx, cond_a) not in by_row or (ridx, cond_b) not in by_row:
                continue
            r_a = by_row[(ridx, cond_a)]
            r_b = by_row[(ridx, cond_b)]
            # Skip only when *all* configured judges are already done for this pair.
            if all((ridx, cond_a, cond_b, jm) in done_keys for jm in cfg.judge.models):
                continue
            try:
                results = judge.compare(
                    context=str(r_a["context"]),
                    question=str(r_a.get("question", "")),
                    response_a=str(r_a["response"]),
                    response_b=str(r_b["response"]),
                )
            except Exception as exc:  # noqa: BLE001
                logger.error("Pairwise %d %s vs %s failed: %s", ridx, cond_a, cond_b, exc)
                continue

            for res in results:
                if (ridx, cond_a, cond_b, res.judge_model) in done_keys:
                    continue
                records.append({
                    "row_id":     ridx,
                    "condition_a": cond_a,
                    "condition_b": cond_b,
                    "judge_model": res.judge_model,
                    "winner":     res.winner,
                    "swap_flag":  bool(res.swap_flag),
                    "parse_error": bool(res.parse_error),
                })
            if len(records) % 20 == 0:
                pd.DataFrame(records).to_csv(pairwise_path, index=False)

    pd.DataFrame(records).to_csv(pairwise_path, index=False)
    logger.info("Pairwise saved → %s (n=%d)", pairwise_path, len(records))


# ─── Stage 4: FActScore consistency check ─────────────────────────────────────

def stage_factscore(cfg: DictConfig, out_dir: Path, outputs_df: pd.DataFrame, comet=None) -> None:
    """
    Compute FActScore-Turbo on every final response.

    Labelled in the analysis as a *consistency check* — gameable by the +FS arm
    because that arm was optimised against this metric (audit R1).
    """
    fs_path = out_dir / "factscore_turbo.csv"
    if fs_path.exists():
        existing = pd.read_csv(fs_path)
        done = set(zip(existing["row_id"].astype(int),
                       existing["condition"].astype(str)))
        records = existing.to_dict(orient="records")
    else:
        done = set()
        records = []

    scorer = FActScoreTurbo(
        model=cfg.factscore.scorer_model,
        temperature=0.0, max_facts=15, batch_verify=True,
    )

    for _, r in outputs_df.iterrows():
        key = (int(r["row_id"]), r["condition"])
        if key in done:
            continue
        try:
            res = scorer.score(response=str(r["response"]), context=str(r["context"]))
        except Exception as exc:  # noqa: BLE001
            logger.error("FActScore failed %s: %s", key, exc)
            continue
        fs_rec = {
            "row_id":      int(r["row_id"]),
            "condition":   r["condition"],
            "factscore":   res.score,
            "n_facts":     res.n_facts,
            "n_supported": res.n_supported,
        }
        records.append(fs_rec)
        if comet is not None:
            comet.log_metric(f"factscore/{r['condition']}", res.score, step=len(records))
        if len(records) % 20 == 0:
            pd.DataFrame(records).to_csv(fs_path, index=False)

    pd.DataFrame(records).to_csv(fs_path, index=False)
    logger.info("FActScore saved → %s (n=%d)", fs_path, len(records))


# ─── Main ─────────────────────────────────────────────────────────────────────

@hydra.main(config_path="../../conf", config_name="guided_generation", version_base=None)
def main(cfg: DictConfig) -> None:
    logger.info("Run config:\n%s", OmegaConf.to_yaml(cfg))

    from comet_ml import Experiment as CometExperiment
    comet = CometExperiment(
        api_key=_COMET_KEY,
        project_name=_COMET_PROJ,
        workspace=_COMET_WS,
        auto_output_logging="simple",
    )
    comet.set_name("guided-generation")
    comet.log_parameters(OmegaConf.to_container(cfg, resolve=True))

    out_dir = _PROJECT_ROOT / cfg.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    outputs_df, _ = stage_generate(cfg, out_dir, comet=comet)
    if outputs_df.empty:
        logger.error("No outputs produced — aborting subsequent stages.")
        comet.end()
        return

    outputs_df = pd.read_parquet(out_dir / "outputs.parquet")

    stage_judge(cfg, out_dir, outputs_df, comet=comet)
    if cfg.get("run_pairwise", True):
        stage_pairwise(cfg, out_dir, outputs_df)
    else:
        logger.info("Skipping pairwise A/B (run_pairwise=false).")
    if cfg.get("run_factscore", True):
        stage_factscore(cfg, out_dir, outputs_df, comet=comet)
    else:
        logger.info("Skipping FActScore consistency (run_factscore=false).")

    # Log final per-condition aggregates
    judge_df = pd.read_csv(out_dir / "judge_scores.csv") if (out_dir / "judge_scores.csv").exists() else pd.DataFrame()
    if not judge_df.empty:
        for cond, grp in judge_df.groupby("condition"):
            comet.log_metrics({
                f"final/{cond}/faithfulness_mean": float(grp["faithfulness"].mean()),
                f"final/{cond}/completeness_mean": float(grp["completeness"].mean()),
                f"final/{cond}/coherence_mean":    float(grp["coherence"].mean()),
                f"final/{cond}/n_responses":       int(len(grp)),
            })
        comet.log_asset(str(out_dir / "judge_scores.csv"), file_name="judge_scores.csv")

    if (out_dir / "analysis.json").exists():
        comet.log_asset(str(out_dir / "analysis.json"), file_name="analysis.json")

    logger.info("Run complete — results at %s", out_dir)
    comet.end()


if __name__ == "__main__":
    main()
