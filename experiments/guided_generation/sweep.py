"""
λ-sweep on the dev split (n=20).

Searches a small grid of (lambda_ll, lambda_fs) values, generates a response per
dev context per cell, and scores each response with the cross-family judge
ensemble.  Picks the cell with the highest mean(faithfulness × completeness)
on the dev set, breaks ties by lower mean wall-time.

Saves:
    sweep_outputs.parquet  one row per (cell, dev row)
    sweep_metrics.csv      per-cell aggregates
    best_lambdas.json      chosen (lambda_ll, lambda_fs) for the final run
    splits.json            dev/test row identities (audit C2)

Wall-time budget on M1 Max: roughly two to three hours; checkpoints are written
after every dev row so the process can be interrupted and resumed safely.
"""
from __future__ import annotations

import json
import logging
import sys
import time
from itertools import product
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
    ConditionConfig,
    build_prompt,
    generate_condition,
    load_generator,
    load_lookback_classifier,
    save_outputs_incremental,
    write_splits_manifest,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

_COMET_KEY  = "WUHdQW2NyhxNGhwB9goVTy3Hi"
_COMET_PROJ = "context-hallucination-prevention"
_COMET_WS   = "vekshinkir"

# Default sweep grids — kept compact to fit the 2–3 h dev budget.
_DEFAULT_LAMBDA_LL_GRID = [0.0, 0.5, 1.0, 2.0]
_DEFAULT_LAMBDA_FS_GRID = [0.0, 0.5, 1.0]


@hydra.main(config_path="../../conf", config_name="guided_generation", version_base=None)
def main(cfg: DictConfig) -> None:
    logger.info("Sweep config:\n%s", OmegaConf.to_yaml(cfg))

    from comet_ml import Experiment as CometExperiment
    comet = CometExperiment(
        api_key=_COMET_KEY,
        project_name=_COMET_PROJ,
        workspace=_COMET_WS,
        auto_output_logging="simple",
    )
    comet.set_name("guided-sweep")
    comet.log_parameters(OmegaConf.to_container(cfg, resolve=True))

    out_dir = _PROJECT_ROOT / cfg.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    sweep_path = out_dir / "sweep_outputs.parquet"

    # ── Splits ────────────────────────────────────────────────────────────────
    dev, test = load_ragtruth_split(
        n_dev=cfg.dataset.n_dev, n_test=cfg.dataset.n_test, seed=cfg.seed,
    )
    write_splits_manifest(dev, test, out_dir)
    logger.info("Loaded dev=%d, test=%d (disjoint).", len(dev), len(test))

    # ── Models ────────────────────────────────────────────────────────────────
    model, tokenizer = load_generator(
        cfg.generator.model, cfg.generator.device, cfg.generator.dtype,
    )
    classifier = load_lookback_classifier(_PROJECT_ROOT / cfg.lookback.classifier_path)
    fs_scorer = FActScoreTurbo(
        model=cfg.factscore.scorer_model,
        temperature=0.0,
        max_facts=10,
        batch_verify=True,
    )
    judge = LLMJudge(models=list(cfg.judge.models), swap_seed=cfg.judge.swap_seed)

    # ── Resume support ────────────────────────────────────────────────────────
    completed: set = set()
    if sweep_path.exists():
        prev = pd.read_parquet(sweep_path)
        if not prev.empty:
            completed = set(zip(prev["row_id"].astype(int),
                                prev["lambda_ll"].astype(float),
                                prev["lambda_fs"].astype(float)))
            logger.info("Resuming sweep — %d cells already complete.", len(completed))

    rows: list[dict] = []
    if sweep_path.exists():
        rows = pd.read_parquet(sweep_path).to_dict(orient="records")

    grid_ll = list(cfg.get("sweep_grid_ll", _DEFAULT_LAMBDA_LL_GRID))
    grid_fs = list(cfg.get("sweep_grid_fs", _DEFAULT_LAMBDA_FS_GRID))
    cells = list(product(grid_ll, grid_fs))
    logger.info("Sweep grid: %d cells over %d dev rows = %d generations",
                len(cells), len(dev), len(cells) * len(dev))

    t_start = time.time()
    for ll, fs in cells:
        cond = ConditionConfig(
            name=f"sweep_LL{ll:g}_FS{fs:g}",
            lambda_ll=float(ll),
            lambda_fs=float(fs),
            use_lookback=(classifier is not None and ll != 0.0),
            use_factscore=(fs != 0.0),
            use_hf_generate=False,
            score_mode=cfg.lookback.get("score_mode", "blend"),
            blend_w=float(cfg.lookback.get("blend_w", 0.5)),
            per_candidate_ll=bool(cfg.lookback.get("per_candidate_ll", True)),
        )
        for ridx, row in dev.iterrows():
            key = (int(ridx), float(ll), float(fs))
            if key in completed:
                continue
            try:
                out = generate_condition(
                    cond=cond,
                    model=model, tokenizer=tokenizer,
                    classifier=classifier, fs_scorer=fs_scorer,
                    prompt=build_prompt(tokenizer, row["context"], row.get("question", "")),
                    context=row["context"],
                    beam_width=cfg.generator.beam_width,
                    max_new_tokens=cfg.generator.max_new_tokens,
                    length_alpha=cfg.generator.length_alpha,
                    seed=cfg.seed,
                )
            except Exception as exc:  # noqa: BLE001 — single-cell failures must not kill the sweep
                logger.error("Cell (LL=%.2f, FS=%.2f) row %s failed: %s", ll, fs, ridx, exc)
                continue

            judge_out = judge.score(
                context=row["context"],
                question=row.get("question", "") or "",
                response=out["response"],
            )

            row_rec = {
                "row_id":         int(ridx),
                "task_type":      row.get("task_type", "Unknown"),
                "lambda_ll":      float(ll),
                "lambda_fs":      float(fs),
                "response":       out["response"],
                "n_response_tokens": int(out["n_response_tokens"]),
                "elapsed_s":      float(out["elapsed_s"]),
                "faithfulness":   float(judge_out.faithfulness),
                "completeness":   float(judge_out.completeness),
                "coherence":      float(judge_out.coherence),
                "judge_std_faith": float(judge_out.judge_std.get("faithfulness", float("nan"))),
                "parse_errors":   ",".join(judge_out.parse_errors),
            }
            rows.append(row_rec)
            save_outputs_incremental(rows, sweep_path)

            step_idx = len(rows)
            comet.log_metrics({
                f"sweep/faithfulness_LL{ll:g}_FS{fs:g}": float(judge_out.faithfulness),
                f"sweep/completeness_LL{ll:g}_FS{fs:g}":  float(judge_out.completeness),
                f"sweep/elapsed_s_LL{ll:g}_FS{fs:g}":     float(out["elapsed_s"]),
            }, step=step_idx)
            comet.log_text(f"[LL={ll:g} FS={fs:g} row={ridx}] {out['response'][:300]}")

            elapsed_total = time.time() - t_start
            logger.info(
                "[%.1fs] LL=%.2f FS=%.2f row=%d  faith=%.2f comp=%.2f resp_tok=%d gen=%.1fs",
                elapsed_total, ll, fs, ridx,
                judge_out.faithfulness, judge_out.completeness,
                out["n_response_tokens"], out["elapsed_s"],
            )

    # ── Aggregate per-cell metrics & pick best ────────────────────────────────
    if not rows:
        logger.error("No sweep rows produced — aborting.")
        return

    df = pd.DataFrame(rows)
    agg = (df.groupby(["lambda_ll", "lambda_fs"])
              .agg(
                  faithfulness_mean=("faithfulness", "mean"),
                  completeness_mean=("completeness", "mean"),
                  coherence_mean=("coherence", "mean"),
                  elapsed_mean=("elapsed_s", "mean"),
                  n=("response", "count"),
              )
              .reset_index())
    agg["score"] = agg["faithfulness_mean"] * agg["completeness_mean"]
    agg = agg.sort_values(by=["score", "elapsed_mean"], ascending=[False, True])
    agg.to_csv(out_dir / "sweep_metrics.csv", index=False)
    logger.info("\nSweep aggregate (sorted):\n%s", agg.to_string(index=False))

    best = agg.iloc[0]
    best_lambdas = {
        "lambda_ll": float(best["lambda_ll"]),
        "lambda_fs": float(best["lambda_fs"]),
        "score":     float(best["score"]),
        "n_dev":     int(best["n"]),
    }
    with open(out_dir / "best_lambdas.json", "w", encoding="utf-8") as fh:
        json.dump(best_lambdas, fh, indent=2)
    logger.info("Best λ on dev: %s", best_lambdas)

    comet.log_parameters({"best_lambda_ll": best_lambdas["lambda_ll"],
                           "best_lambda_fs": best_lambdas["lambda_fs"]})
    comet.log_asset(str(out_dir / "sweep_metrics.csv"), file_name="sweep_metrics.csv")
    comet.end()


if __name__ == "__main__":
    main()
