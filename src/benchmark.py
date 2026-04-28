"""
Benchmark utilities for contextual-hallucination evaluation.

Provides:
- Dataset loaders (RAGTruth, HalluMix) normalised to a common schema.
- Response generator (local Ollama model).
- Benchmark runner (FActScore-Turbo over a DataFrame).
- Metrics computation (AUC-ROC, F1, Precision, Recall, optimal threshold).
"""
from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from datasets import load_dataset
from tqdm.auto import tqdm

from .factscore_turbo import FActScoreTurbo

logger = logging.getLogger(__name__)

# ─── Internal helpers ─────────────────────────────────────────────────────────

def _normalise_context(val) -> str:
    """Convert any context representation to a plain string."""
    if isinstance(val, str):
        return val.strip()
    if isinstance(val, list):
        parts = []
        for item in val:
            if isinstance(item, str):
                parts.append(item.strip())
            elif isinstance(item, dict):
                text = item.get("text") or item.get("passage") or item.get("content") or str(item)
                parts.append(str(text).strip())
            else:
                parts.append(str(item).strip())
        return "\n\n".join(p for p in parts if p)
    if isinstance(val, dict):
        text = val.get("text") or val.get("passage") or val.get("content") or str(val)
        return str(text).strip()
    return str(val).strip()


def _extract_hallucination_label_ragtruth(labels) -> bool:
    """
    Derive a binary hallucination label from RAGTruth's ``labels`` field.

    Handles three schemas observed in the dataset:
    - bool / numeric: direct cast.
    - list of span dicts: hallucinated iff non-empty.
    - dict with 'spans'/'hallucinations' keys: hallucinated iff non-empty span list.
    - dict with integer flag values (e.g. {'evident_conflict': 1, 'baseless_info': 0}):
      hallucinated iff any flag > 0.
    """
    if isinstance(labels, bool):
        return labels
    if isinstance(labels, (int, float)):
        return bool(labels)
    if isinstance(labels, list):
        return len(labels) > 0
    if isinstance(labels, dict):
        spans = labels.get("spans") or labels.get("hallucinations")
        if spans is not None:
            return len(spans) > 0
        # Fallback: treat all dict values as numeric flags.
        return any(bool(v) for v in labels.values())
    return False


# ─── Dataset loaders ──────────────────────────────────────────────────────────

# Common output schema:
#   context         str   – source passage(s)
#   question        str   – input question (may be empty for summarisation)
#   response        str   – model-generated answer to evaluate
#   is_hallucinated bool  – ground-truth hallucination label
#   task_type       str   – e.g. 'QA', 'Summarization', 'Data2Text', 'Mixed'
#   gen_model       str   – model that generated the response ('Unknown' if missing)
#   source          str   – dataset identifier ('ragtruth' / 'hallumix')

_SCHEMA = ["context", "question", "response", "is_hallucinated", "task_type", "gen_model", "source"]


def load_ragtruth(
    n_samples: Optional[int] = None,
    task_filter: Optional[str] = None,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Load and normalise the RAGTruth dataset.

    Parameters
    ----------
    n_samples : int, optional
        Stratified sample size (balanced across hallucinated / faithful).
    task_filter : str, optional
        Keep only rows whose task_type contains this string (e.g. 'QA').
    seed : int
        Random seed for sampling.
    """
    logger.info("Loading RAGTruth …")
    ds = load_dataset("wandb/RAGTruth-processed")
    split = list(ds.keys())[0]
    df = ds[split].to_pandas()
    logger.info("RAGTruth raw shape: %s", df.shape)

    # ── context ──────────────────────────────────────────────────────────────
    ctx_candidates = [c for c in df.columns if any(k in c.lower() for k in ("source", "context", "passage", "document"))]
    ctx_col = ctx_candidates[0] if ctx_candidates else None
    if ctx_col:
        df["context"] = df[ctx_col].apply(_normalise_context)
    else:
        raise ValueError(f"Cannot find context column. Available: {list(df.columns)}")

    # ── question ─────────────────────────────────────────────────────────────
    q_candidates = [c for c in df.columns if any(k in c.lower() for k in ("question", "query", "input"))]
    df["question"] = df[q_candidates[0]] if q_candidates else ""

    # ── response ─────────────────────────────────────────────────────────────
    resp_candidates = [c for c in df.columns if any(k in c.lower() for k in ("response", "answer", "output", "generated"))]
    resp_col = resp_candidates[0] if resp_candidates else None
    if resp_col:
        df["response"] = df[resp_col].fillna("").astype(str)
    else:
        raise ValueError(f"Cannot find response column. Available: {list(df.columns)}")

    # ── hallucination label ───────────────────────────────────────────────────
    if "has_hallucination" in df.columns:
        df["is_hallucinated"] = df["has_hallucination"].astype(bool)
    elif "labels" in df.columns:
        df["is_hallucinated"] = df["labels"].apply(_extract_hallucination_label_ragtruth)
    elif "hallucination_labels_processed" in df.columns:
        df["is_hallucinated"] = df["hallucination_labels_processed"].apply(
            _extract_hallucination_label_ragtruth
        )
    else:
        raise ValueError("Cannot find hallucination label. Available: " + str(list(df.columns)))

    # ── task type ─────────────────────────────────────────────────────────────
    task_candidates = [c for c in df.columns if "task" in c.lower()]
    df["task_type"] = df[task_candidates[0]] if task_candidates else "Unknown"

    # ── generating model ──────────────────────────────────────────────────────
    model_candidates = [c for c in df.columns if c.lower() == "model" or "model_name" in c.lower()]
    df["gen_model"] = df[model_candidates[0]] if model_candidates else "Unknown"

    df["source"] = "ragtruth"

    # ── filter & clean ────────────────────────────────────────────────────────
    if task_filter:
        df = df[df["task_type"].str.contains(task_filter, case=False, na=False)]
    df = df.dropna(subset=["context", "response"])
    df = df[df["context"].str.len() > 30]
    df = df[df["response"].str.len() > 10]
    df = df.reset_index(drop=True)

    # ── stratified sample ─────────────────────────────────────────────────────
    if n_samples:
        pos = df[df["is_hallucinated"]]
        neg = df[~df["is_hallucinated"]]
        n_each = min(n_samples // 2, len(pos), len(neg))
        df = pd.concat([
            pos.sample(n_each, random_state=seed),
            neg.sample(n_each, random_state=seed),
        ]).sample(frac=1, random_state=seed).reset_index(drop=True)
        logger.info("Sampled %d samples (balanced)", len(df))

    logger.info(
        "RAGTruth ready: %d rows | hallucinated=%d (%.1f%%)",
        len(df), df["is_hallucinated"].sum(), 100 * df["is_hallucinated"].mean(),
    )
    return df[_SCHEMA]


def load_hallumix(
    n_samples: Optional[int] = None,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Load and normalise the HalluMix dataset (multi-domain cross-evaluation).

    Parameters
    ----------
    n_samples : int, optional
        Random sample size.
    seed : int
        Random seed.
    """
    logger.info("Loading HalluMix …")
    ds = load_dataset("quotientai/HalluMix", trust_remote_code=True)
    split = list(ds.keys())[0]
    df = ds[split].to_pandas()
    logger.info("HalluMix raw shape: %s", df.shape)

    # ── context ───────────────────────────────────────────────────────────────
    doc_col = next(
        (c for c in df.columns if c.lower() in ("documents", "context", "passages", "chunks", "source")),
        None,
    )
    df["context"] = df[doc_col].apply(_normalise_context) if doc_col else ""

    # ── response ──────────────────────────────────────────────────────────────
    ans_col = next(
        (c for c in df.columns if c.lower() in ("answer", "response", "hypothesis", "output", "text")),
        None,
    )
    df["response"] = df[ans_col].fillna("").astype(str) if ans_col else ""

    # ── question ──────────────────────────────────────────────────────────────
    q_col = next((c for c in df.columns if "question" in c.lower() or "query" in c.lower()), None)
    df["question"] = df[q_col].fillna("").astype(str) if q_col else ""

    # ── label ─────────────────────────────────────────────────────────────────
    lbl_col = next(
        (c for c in df.columns if "hallucin" in c.lower() or c.lower() == "label"),
        None,
    )
    df["is_hallucinated"] = df[lbl_col].astype(bool) if lbl_col else False

    # ── task / model ──────────────────────────────────────────────────────────
    task_col = next((c for c in df.columns if c.lower() in ("task", "task_type", "domain")), None)
    df["task_type"] = df[task_col].fillna("Mixed").astype(str) if task_col else "Mixed"
    df["gen_model"] = "Unknown"
    df["source"] = "hallumix"

    # ── clean ─────────────────────────────────────────────────────────────────
    df = df.dropna(subset=["context", "response"])
    df = df[df["context"].str.len() > 30]
    df = df[df["response"].str.len() > 10]
    df = df.reset_index(drop=True)

    if n_samples:
        df = df.sample(min(n_samples, len(df)), random_state=seed).reset_index(drop=True)

    logger.info(
        "HalluMix ready: %d rows | hallucinated=%d (%.1f%%)",
        len(df), df["is_hallucinated"].sum(), 100 * df["is_hallucinated"].mean(),
    )
    return df[_SCHEMA]


# ─── Response generator ───────────────────────────────────────────────────────

_GEN_SYSTEM = (
    "You are a helpful assistant. Answer questions based ONLY on the provided context. "
    "If the answer is not present in the context, respond: "
    "\"Based on the provided context, I cannot find information about this.\" "
    "Be concise and do not add information beyond what is in the context."
)

_GEN_PROMPT = """\
Context:
{context}

Question: {question}

Answer:"""

_SUMM_PROMPT = """\
Context:
{context}

Provide a concise summary of the above context:"""


def generate_responses(
    df: pd.DataFrame,
    model: str = "qwen2.5:7b",
    max_new_tokens: int = 256,
    n_samples: Optional[int] = None,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate responses for each sample using a local Ollama model.

    Returns the input DataFrame with an added ``generated_response`` column.
    """
    try:
        import ollama
    except ImportError as exc:
        raise ImportError("Install ollama: pip install ollama") from exc

    if n_samples:
        df = df.sample(min(n_samples, len(df)), random_state=seed).reset_index(drop=True)

    responses: list[str] = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Generating [{model}]"):
        ctx = str(row["context"])[:2_500]
        question = str(row.get("question", "")).strip()

        prompt = _GEN_PROMPT.format(context=ctx, question=question) if question \
            else _SUMM_PROMPT.format(context=ctx)

        try:
            resp = ollama.chat(
                model=model,
                messages=[
                    {"role": "system", "content": _GEN_SYSTEM},
                    {"role": "user",   "content": prompt},
                ],
                options={"temperature": 0.1, "num_predict": max_new_tokens},
            )
            responses.append(resp["message"]["content"].strip())
        except Exception as exc:
            logger.error("Generation failed for row %s: %s", _, exc)
            responses.append("")

    result = df.copy()
    result["generated_response"] = responses
    return result


# ─── Benchmark runner ─────────────────────────────────────────────────────────

def run_factscore_benchmark(
    df: pd.DataFrame,
    scorer: FActScoreTurbo,
    response_col: str = "response",
    n_samples: Optional[int] = None,
    save_path: Optional[Path] = None,
    seed: int = 42,
) -> tuple[pd.DataFrame, dict]:
    """
    Run FActScore-Turbo on every sample in *df* and compute detection metrics.

    Parameters
    ----------
    df           : DataFrame with columns ``context``, *response_col*, ``is_hallucinated``.
    scorer       : Configured FActScoreTurbo instance.
    response_col : Column name of the response to evaluate.
    n_samples    : Evaluate only a random subset of *n_samples*.
    save_path    : If given, saves ``{save_path}.csv`` and ``{save_path}.json``.
    seed         : Random seed for sampling.

    Returns
    -------
    results_df : Original DataFrame enriched with scoring columns.
    metrics    : Dict with AUC-ROC, F1, precision, recall, optimal threshold, etc.
    """
    if n_samples:
        df = df.sample(min(n_samples, len(df)), random_state=seed).reset_index(drop=True)

    records: list[dict] = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="FActScore evaluation"):
        result = scorer.score(
            response=str(row[response_col]),
            context=str(row["context"]),
        )
        records.append({
            "factscore":   result.score,
            "n_facts":     result.n_facts,
            "n_supported": result.n_supported,
            "facts":       result.facts,
            "supported":   result.supported,
            "score_error": result.error,
        })

    results_df = df.copy().reset_index(drop=True)
    results_df = pd.concat([results_df, pd.DataFrame(records)], axis=1)

    metrics = compute_metrics(results_df)

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        # Save CSV without list columns (they don't serialise well)
        csv_df = results_df.drop(columns=["facts", "supported"], errors="ignore")
        csv_df.to_csv(save_path.with_suffix(".csv"), index=False)
        with open(save_path.with_suffix(".json"), "w", encoding="utf-8") as fh:
            json.dump(metrics, fh, indent=2, ensure_ascii=False)
        logger.info("Results saved → %s", save_path.parent)

    return results_df, metrics


# ─── Metrics ──────────────────────────────────────────────────────────────────

def compute_metrics(df: pd.DataFrame, threshold: float = 0.5) -> dict:
    """
    Compute hallucination-detection metrics.

    FActScore-Turbo predicts hallucination when ``factscore < threshold``.

    Parameters
    ----------
    df        : Must contain ``factscore`` (float) and ``is_hallucinated`` (bool).
    threshold : Decision boundary.  Applied as ``factscore < threshold`` → hallucinated.

    Returns
    -------
    metrics : Dict with keys: roc_auc, f1, precision, recall, accuracy, optimal_threshold, …
    """
    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        precision_recall_curve,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    valid = df.dropna(subset=["factscore", "is_hallucinated"]).copy()
    if valid.empty:
        return {"error": "No valid samples after dropping NaN"}

    y_true  = valid["is_hallucinated"].astype(int).values
    # Higher hallucination score = lower FActScore
    y_score = (1 - valid["factscore"]).values
    y_pred  = (valid["factscore"] < threshold).astype(int).values

    metrics: dict = {
        "n_samples":             int(len(valid)),
        "n_hallucinated_gt":     int(y_true.sum()),
        "n_faithful_gt":         int(len(y_true) - y_true.sum()),
        "hallucination_rate_gt": float(y_true.mean()),
        "hallucination_rate_pred": float(y_pred.mean()),
        "threshold":             threshold,
        "avg_factscore":         float(valid["factscore"].mean()),
        "std_factscore":         float(valid["factscore"].std()),
        "median_factscore":      float(valid["factscore"].median()),
        "avg_n_facts":           float(valid["n_facts"].mean()) if "n_facts" in valid.columns else None,
    }

    # ROC-AUC
    try:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_score))
    except ValueError:
        metrics["roc_auc"] = None

    # F1 / Precision / Recall at the chosen threshold
    for average in ("binary", "macro"):
        sfx = "" if average == "binary" else "_macro"
        kw  = dict(average=average, zero_division=0)
        metrics[f"f1{sfx}"]        = float(f1_score(y_true, y_pred, **kw))
        metrics[f"precision{sfx}"] = float(precision_score(y_true, y_pred, **kw))
        metrics[f"recall{sfx}"]    = float(recall_score(y_true, y_pred, **kw))

    metrics["accuracy"] = float(accuracy_score(y_true, y_pred))

    # Optimal threshold (max F1 on PR curve)
    try:
        prec, rec, thresholds = precision_recall_curve(y_true, y_score)
        f1_arr  = 2 * prec * rec / (prec + rec + 1e-9)
        best_i  = int(f1_arr.argmax())
        metrics["optimal_threshold"] = float(1 - thresholds[best_i])   # back to FActScore scale
        metrics["optimal_f1"]        = float(f1_arr[best_i])
        metrics["optimal_precision"] = float(prec[best_i])
        metrics["optimal_recall"]    = float(rec[best_i])
    except Exception:
        pass

    return metrics
