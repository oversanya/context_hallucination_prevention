"""
Shared infrastructure for the guided beam-search experiments.

Contains the bits used by *both* sweep.py and run.py:

* generator + classifier loading
* prompt construction (RAGTruth QA / summarisation)
* a single ``generate(condition)`` entry point that wraps :func:`guided_beam_search`
  for the four custom-loop conditions and HuggingFace ``model.generate`` for the
  ``vanilla_hf`` reference (audit R3)
* incremental parquet checkpointing so a crash mid-overnight loses at most
  ``checkpoint_every`` rows (audit R5)
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src import FActScoreTurbo, guided_beam_search
from src.guided_beam_search import GuidedGenerationResult
from src.lookback_lens import LookbackLensClassifier

logger = logging.getLogger(__name__)


# ─── Prompt ───────────────────────────────────────────────────────────────────

_SYSTEM = (
    "You are a helpful assistant. Answer questions based ONLY on the provided "
    "context. If the answer is not present in the context, say so explicitly. "
    "Be concise. Do not add information not in the context."
)

_USER_QA = (
    "Context:\n\"\"\"\n{context}\n\"\"\"\n\n"
    "Question: {question}\n\nAnswer:"
)

_USER_SUMMARY = (
    "Context:\n\"\"\"\n{context}\n\"\"\"\n\n"
    "Provide a concise summary of the above context:"
)


def build_prompt(tokenizer, context: str, question: str) -> str:
    """Build a fully-formatted chat prompt suitable for Qwen-Instruct models."""
    user = _USER_QA.format(context=context, question=question) if question.strip() \
        else _USER_SUMMARY.format(context=context)
    messages = [
        {"role": "system", "content": _SYSTEM},
        {"role": "user",   "content": user},
    ]
    # Use the chat template only when one is actually configured on the tokenizer.
    # Older base models (e.g. OPT) expose `apply_chat_template` but have no
    # `chat_template` attribute set, in which case the call raises ValueError.
    if getattr(tokenizer, "chat_template", None):
        try:
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
        except (ValueError, TypeError):
            pass
    return _SYSTEM + "\n\n" + user + "\n"


# ─── Generator ────────────────────────────────────────────────────────────────

def load_generator(model_name: str, device: str, dtype: str):
    """Load a HF causal LM + tokenizer in eval mode with attentions enabled."""
    torch_dtype = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }.get(dtype, torch.float32)

    logger.info("Loading tokenizer %s", model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info("Loading model %s on %s (%s)", model_name, device, dtype)
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            attn_implementation="eager",   # required for output_attentions=True on MPS
        ).to(device)
    except (TypeError, ValueError):
        # Older architectures (e.g. OPT) don't accept attn_implementation.
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch_dtype,
        ).to(device)
    model.eval()
    logger.info("Model loaded — %d layers, %d heads",
                model.config.num_hidden_layers, model.config.num_attention_heads)
    return model, tokenizer


def load_lookback_classifier(path: str | Path) -> Optional[LookbackLensClassifier]:
    """Load a trained classifier; return ``None`` if the file is missing."""
    p = Path(path)
    if not p.exists():
        logger.warning("Classifier not found at %s — LL term will be disabled.", p)
        return None
    return LookbackLensClassifier.load(p)


# ─── Per-condition entry point ────────────────────────────────────────────────

@dataclass
class ConditionConfig:
    """Static per-condition settings (no per-row state)."""

    name: str          # one of: vanilla_hf, vanilla_loop, lookback, factscore, combined
    lambda_ll: float
    lambda_fs: float
    use_lookback:  bool
    use_factscore: bool
    use_hf_generate: bool   # True only for vanilla_hf
    score_mode:      str  = "blend"
    blend_w:         float = 0.5
    per_candidate_ll: bool = True


def make_conditions(
    lambda_ll_best: float,
    lambda_fs_best: float,
    score_mode: str = "blend",
    blend_w: float = 0.5,
    per_candidate_ll: bool = True,
) -> list[ConditionConfig]:
    """Return the five conditions in the order they will be reported."""
    return [
        ConditionConfig("vanilla_hf",   0.0,            0.0,            False, False, True,  score_mode, blend_w, per_candidate_ll),
        ConditionConfig("vanilla_loop", 0.0,            0.0,            False, False, False, score_mode, blend_w, per_candidate_ll),
        ConditionConfig("lookback",     lambda_ll_best, 0.0,            True,  False, False, score_mode, blend_w, per_candidate_ll),
        ConditionConfig("factscore",    0.0,            lambda_fs_best, False, True,  False, score_mode, blend_w, per_candidate_ll),
        ConditionConfig("combined",     lambda_ll_best, lambda_fs_best, True,  True,  False, score_mode, blend_w, per_candidate_ll),
    ]


def _hf_vanilla_generate(
    model,
    tokenizer,
    prompt: str,
    *,
    beam_width: int,
    max_new_tokens: int,
    length_alpha: float,
    seed: int,
) -> tuple[str, list, float]:
    """Reference HF beam search (audit R3)."""
    torch.manual_seed(seed)
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
    t0 = time.time()
    with torch.no_grad():
        out = model.generate(
            **inputs,
            num_beams=beam_width,
            do_sample=False,
            length_penalty=length_alpha,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            return_dict_in_generate=False,
        )
    elapsed = time.time() - t0
    response_ids = out[0, inputs.input_ids.shape[1]:].tolist()
    text = tokenizer.decode(response_ids, skip_special_tokens=True)
    return text, response_ids, float(elapsed)


def generate_condition(
    cond: ConditionConfig,
    *,
    model,
    tokenizer,
    classifier: Optional[LookbackLensClassifier],
    fs_scorer:  Optional[FActScoreTurbo],
    prompt: str,
    context: str,
    beam_width: int,
    max_new_tokens: int,
    length_alpha: float,
    seed: int,
    log_per_step: bool = False,
    fs_neutral: float = 0.5,
) -> dict:
    """
    Run one condition on one (prompt, context) pair and return a flat dict
    with the response, latency, and per-step log (if requested).
    """
    if cond.use_hf_generate:
        text, ids, elapsed = _hf_vanilla_generate(
            model=model, tokenizer=tokenizer, prompt=prompt,
            beam_width=beam_width, max_new_tokens=max_new_tokens,
            length_alpha=length_alpha, seed=seed,
        )
        return {
            "condition":       cond.name,
            "response":        text,
            "response_token_ids": ids,
            "n_response_tokens":  len(ids),
            "elapsed_s":       elapsed,
            "lambda_ll":       cond.lambda_ll,
            "lambda_fs":       cond.lambda_fs,
            "step_log":        [],
            "final_score":     float("nan"),
        }

    result: GuidedGenerationResult = guided_beam_search(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        context=context,
        lookback_classifier=classifier if cond.use_lookback else None,
        factscore_scorer=fs_scorer  if cond.use_factscore else None,
        lambda_ll=cond.lambda_ll,
        lambda_fs=cond.lambda_fs,
        length_alpha=length_alpha,
        beam_width=beam_width,
        max_new_tokens=max_new_tokens,
        seed=seed,
        log_per_step=log_per_step,
        fs_neutral=fs_neutral,
        score_mode=cond.score_mode,
        blend_w=cond.blend_w,
        per_candidate_ll=cond.per_candidate_ll,
    )
    return {
        "condition":           cond.name,
        "response":            result.text,
        "response_token_ids":  result.response_token_ids,
        "n_response_tokens":   len(result.response_token_ids),
        "elapsed_s":           result.elapsed_s,
        "lambda_ll":           cond.lambda_ll,
        "lambda_fs":           cond.lambda_fs,
        "step_log":            [asdict(r) for r in result.step_log],
        "final_score":         result.final_score,
    }


# ─── Incremental checkpointing ────────────────────────────────────────────────

def save_outputs_incremental(rows: list, out_path: Path) -> None:
    """Persist a parquet snapshot atomically (write to .tmp then rename)."""
    if not rows:
        return
    df = pd.DataFrame(rows)
    tmp = out_path.with_suffix(".tmp.parquet")
    df.to_parquet(tmp)
    tmp.replace(out_path)


def load_completed_keys(out_path: Path) -> set[tuple[int, str]]:
    """Return the set of (row_id, condition) tuples already written to parquet."""
    if not out_path.exists():
        return set()
    df = pd.read_parquet(out_path)
    if df.empty:
        return set()
    return set(zip(df["row_id"].astype(int), df["condition"].astype(str)))


# ─── Splits manifest ──────────────────────────────────────────────────────────

def write_splits_manifest(
    dev: pd.DataFrame, test: pd.DataFrame, out_dir: Path,
) -> None:
    """Persist a JSON manifest of dev/test row identities (audit C2)."""
    manifest = {
        "n_dev":          int(len(dev)),
        "n_test":         int(len(test)),
        "dev_responses":  dev["response"].astype(str).tolist(),
        "test_responses": test["response"].astype(str).tolist(),
    }
    with open(out_dir / "splits.json", "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, ensure_ascii=False, indent=2)


# ─── Lightweight float casting ────────────────────────────────────────────────

def to_jsonable(x):
    """Make numpy / torch types JSON-serialisable."""
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    if torch.is_tensor(x):
        return x.tolist()
    return x
