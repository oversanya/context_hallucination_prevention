"""
Guided beam search for hallucination-mitigated text generation.

Implements a custom beam-search loop that augments the standard cumulative
log-probability with two faithfulness signals:

* **Lookback Lens** (token-level): a logistic-regression classifier over per-head
  lookback ratios extracted from the generator's own attention map.  The score
  is computed from the attentions already produced by the forward pass, with
  no additional inference cost.
* **FActScore-Turbo** (sentence-level): an Ollama-backed atomic-fact scorer that
  is triggered only when a beam ends a sentence.  When no atomic facts can be
  extracted the scorer returns a neutral 0.5 to suppress the "say nothing"
  degenerate strategy (see audit C4 of the guided-generation plan).

The reranking score for a candidate beam at length ``L`` is

    score = (sum_logp / L**alpha) + lambda_ll * LL + 1[sent_end] * lambda_fs * FS

with ``alpha`` the GNMT length-normalisation exponent (default 0.7).  Setting
``lambda_ll = lambda_fs = 0`` recovers length-normalised vanilla beam search.
"""
from __future__ import annotations

import logging
import random
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

from .factscore_turbo import FActScoreTurbo
from .lookback_lens import LookbackLensClassifier, LookbackRatioExtractor

logger = logging.getLogger(__name__)


# ─── Sentence boundary detection ──────────────────────────────────────────────

_SENTENCE_END_CHARS = (".", "!", "?")


def _is_sentence_end(decoded_token: str) -> bool:
    """Return True iff a decoded token contains a sentence-ending punctuation."""
    return any(ch in decoded_token for ch in _SENTENCE_END_CHARS)


# ─── Per-step log entries ─────────────────────────────────────────────────────

@dataclass
class StepRecord:
    """Per-step record for the §D notebook (rejected-token highlighting)."""

    step:           int
    parent_beam_id: int
    token_id:       int
    token_text:     str
    token_logp:     float
    ll_score:       float
    fs_score:       float
    sentence_end:   bool
    rerank_score:   float
    kept:           bool


# ─── Result container ─────────────────────────────────────────────────────────

@dataclass
class GuidedGenerationResult:
    """Output of :func:`guided_beam_search`."""

    text:                str
    response_token_ids:  list
    final_score:         float
    n_steps:             int
    elapsed_s:           float
    config:              dict
    step_log:            list = field(default_factory=list)   # list[StepRecord]
    all_beams_text:      list = field(default_factory=list)   # final beam texts
    # Sanity counters (audit-mandated for the design-defect fix):
    n_ll_reorders:       int  = 0   # steps where LL flipped the top-W set
    n_sentence_end_steps: int = 0   # steps with any sentence-end candidate
    n_total_steps:       int  = 0   # total decoding steps


# ─── Determinism ──────────────────────────────────────────────────────────────

def _seed_all(seed: int) -> None:
    """Seed every relevant RNG so the generator is reproducible (audit R7)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        try:
            torch.mps.manual_seed(seed)  # type: ignore[attr-defined]
        except (AttributeError, RuntimeError):
            # Older PyTorch versions lack mps.manual_seed; fall back silently.
            pass
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ─── Internal beam state ──────────────────────────────────────────────────────

@dataclass
class _Beam:
    """Mutable per-beam state during the search."""

    token_ids:            list           # full prompt + response tokens (ints)
    sum_logp:             float          # cumulative log-prob (response only)
    response_len:         int            # number of generated response tokens
    finished:             bool
    sentence_buffer:      str            # text since last sentence boundary
    last_sentence_fs:     float          # most recent FS score (carried until next boundary)
    parent_id:            int            # id within previous step (for §D log)


# ─── Main entry point ─────────────────────────────────────────────────────────

def guided_beam_search(
    model,
    tokenizer,
    prompt: str,
    context: str,
    *,
    lookback_classifier: Optional[LookbackLensClassifier] = None,
    factscore_scorer:    Optional[FActScoreTurbo]         = None,
    lambda_ll:           float = 0.0,
    lambda_fs:           float = 0.0,
    length_alpha:        float = 0.7,
    beam_width:          int   = 4,
    max_new_tokens:      int   = 128,
    eos_token_id:        Optional[int] = None,
    seed:                int   = 42,
    log_per_step:        bool  = False,
    fs_neutral:          float = 0.5,
    score_mode:          str   = "additive",   # "additive" or "blend"
    blend_w:             float = 0.5,
    per_candidate_ll:    bool  = True,
) -> GuidedGenerationResult:
    """
    Run guided beam search and return the highest-scoring beam.

    Parameters
    ----------
    model, tokenizer
        A HuggingFace causal LM and its tokenizer.  The model must expose
        ``output_attentions=True`` in its forward pass for the LL term to work.
    prompt
        The full text fed to the model (e.g. system + user message string).
    context
        The grounding passage(s).  Used by the FActScore-Turbo scorer for
        sentence-level verification, and to delimit *context* vs *response*
        positions in the attention matrix when computing the lookback ratio.
        Internally we treat the entire prompt as the context (n_ctx = prompt
        token count), since lookback is a property of the response distribution
        relative to the prompt.
    lookback_classifier
        Trained classifier; ``None`` disables the LL term.  Its ``predict_proba``
        returns ``P(hallucinated)``; the reranking signal is ``1 - P``.
    factscore_scorer
        Configured FActScoreTurbo instance; ``None`` disables the FS term.
    lambda_ll, lambda_fs
        Non-negative weights of the two faithfulness signals.
    length_alpha
        GNMT length-normalisation exponent in ``(sum_logp / L**alpha)``.  Set to
        0 to recover unnormalised log-probability; 1.0 for full per-token mean.
    beam_width
        Beams retained at each step.
    max_new_tokens
        Cap on response length (response tokens, not including prompt).
    eos_token_id
        End-of-sequence id; defaults to ``tokenizer.eos_token_id``.
    seed
        Determinism seed; all RNG state is reset at entry.
    log_per_step
        If True, the result includes a :class:`StepRecord` for every kept and
        rejected candidate.  Off by default to keep memory bounded.
    fs_neutral
        Sentence-level FS score returned when no atomic facts can be extracted.
    score_mode
        ``"additive"``: ``score = norm_logp + λ_LL·LL + 1[end]·λ_FS·FS``.
        ``"blend"``: convex blend of standardised log-prob and LL,
        ``score = (1-w)·z(norm_logp) + w·z(LL) + 1[end]·λ_FS·FS`` (audit-recommended,
        removes magnitude mismatch between log-prob (nats) and LL (∈[0,1])).
    blend_w
        Convex blend weight in ``[0, 1]`` when ``score_mode='blend'``.
    per_candidate_ll
        When True (default and audit-mandated), recompute the LL feature
        *with the candidate token appended* so that LL differentiates
        candidates within a parent beam (not just across beams).
    """
    _seed_all(seed)
    t0 = time.time()

    if eos_token_id is None:
        eos_token_id = tokenizer.eos_token_id

    device = next(model.parameters()).device

    # Tokenise prompt once.
    prompt_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids[0].tolist()
    n_ctx = len(prompt_ids)

    # Optional extractor — needed when LL term is active in either scoring mode.
    # In blend mode the LL signal is always active when a classifier is supplied
    # (blend_w > 0 is not checked here; that's a tuning decision).
    extractor: Optional[LookbackRatioExtractor] = None
    ll_active = lookback_classifier is not None and (
        score_mode == "blend" or lambda_ll != 0.0
    )
    if ll_active:
        extractor = LookbackRatioExtractor.from_dims(
            n_layers=model.config.num_hidden_layers,
            n_heads=model.config.num_attention_heads,
        )

    # Initial beam.
    beams: list[_Beam] = [_Beam(
        token_ids=list(prompt_ids),
        sum_logp=0.0,
        response_len=0,
        finished=False,
        sentence_buffer="",
        last_sentence_fs=fs_neutral,
        parent_id=0,
    )]

    step_log: list[StepRecord] = []
    n_ll_reorders = 0
    n_sentence_end_steps = 0
    n_total_steps = 0

    for step in range(max_new_tokens):
        n_total_steps += 1
        # If every beam is finished, stop.
        if all(b.finished for b in beams):
            break

        candidates: list[dict] = []

        for beam_id, beam in enumerate(beams):
            if beam.finished:
                # Carry the finished beam forward as an "EOS-only" candidate so
                # it can still compete on rerank score against unfinished beams.
                candidates.append({
                    "parent_id":   beam_id,
                    "tok_id":      eos_token_id,
                    "tok_logp":    0.0,
                    "ll_score":    0.0,
                    "fs_score":    beam.last_sentence_fs,
                    "sentence_end": False,
                    "new_logp":    beam.sum_logp,
                    "new_resp_len": beam.response_len,
                    "new_token_text": "",
                    "new_sentence_buffer": beam.sentence_buffer,
                    "finishes":    True,
                })
                continue

            input_ids = torch.tensor(
                [beam.token_ids], dtype=torch.long, device=device,
            )
            with torch.no_grad():
                out = model(
                    input_ids=input_ids,
                    output_attentions=(extractor is not None and not per_candidate_ll),
                    use_cache=False,
                )
            logits = out.logits[0, -1, :]
            log_probs = F.log_softmax(logits, dim=-1)
            topk_logp, topk_idx = log_probs.topk(beam_width)

            # Parent-beam LL fallback (used only when per_candidate_ll=False).
            parent_ll = 0.0
            if extractor is not None and not per_candidate_ll and beam.response_len > 0:
                features = extractor.compute_from_attentions(
                    out.attentions, n_ctx=n_ctx, n_resp=beam.response_len,
                )
                p_hall = float(
                    lookback_classifier.predict_proba(features.reshape(1, -1))[0]
                )
                parent_ll = 1.0 - p_hall

            for k in range(beam_width):
                tok_id   = int(topk_idx[k].item())
                tok_logp = float(topk_logp[k].item())
                new_logp     = beam.sum_logp + tok_logp
                new_resp_len = beam.response_len + 1
                new_token_text = tokenizer.decode([tok_id])
                sentence_done  = _is_sentence_end(new_token_text)

                # ── Per-candidate LL (audit-mandated) ────────────────────────
                # Recompute the lookback feature with the candidate token
                # appended so that LL differentiates *tokens within a beam*
                # (not just across beams).
                ll_score = 0.0
                if extractor is not None and per_candidate_ll:
                    candidate_ids = beam.token_ids + [tok_id]
                    cand_inp = torch.tensor([candidate_ids], dtype=torch.long, device=device)
                    with torch.no_grad():
                        cand_out = model(
                            input_ids=cand_inp,
                            output_attentions=True,
                            use_cache=False,
                        )
                    feats = extractor.compute_from_attentions(
                        cand_out.attentions, n_ctx=n_ctx, n_resp=new_resp_len,
                    )
                    p_hall = float(
                        lookback_classifier.predict_proba(feats.reshape(1, -1))[0]
                    )
                    ll_score = 1.0 - p_hall
                elif extractor is not None:
                    ll_score = parent_ll

                # FS term — fires only at sentence boundaries.
                fs_score = beam.last_sentence_fs
                if sentence_done and factscore_scorer is not None and lambda_fs != 0.0:
                    sent_text = (beam.sentence_buffer + new_token_text).strip()
                    fs_result = factscore_scorer.score_sentence(
                        sentence=sent_text,
                        context=context,
                        neutral_score_when_no_facts=fs_neutral,
                    )
                    fs_score = float(fs_result.score) if not np.isnan(fs_result.score) else fs_neutral

                new_sentence_buffer = "" if sentence_done else beam.sentence_buffer + new_token_text

                candidates.append({
                    "parent_id":          beam_id,
                    "tok_id":             tok_id,
                    "tok_logp":           tok_logp,
                    "ll_score":           ll_score,
                    "fs_score":           fs_score,
                    "sentence_end":       sentence_done,
                    "new_logp":           new_logp,
                    "new_resp_len":       new_resp_len,
                    "new_token_text":     new_token_text,
                    "new_sentence_buffer": new_sentence_buffer,
                    "finishes":           tok_id == eos_token_id,
                })

        # ── Sentence-end tracker ─────────────────────────────────────────
        if any(c["sentence_end"] for c in candidates if not c.get("finishes")):
            n_sentence_end_steps += 1

        # ── Compute rerank score ──────────────────────────────────────────
        for c in candidates:
            length = max(c["new_resp_len"], 1)
            c["_norm_logp"] = c["new_logp"] / (length ** length_alpha) if length_alpha > 0 else c["new_logp"]

        if score_mode == "blend" and extractor is not None:
            # Convex blend: standardise both signals per step, then mix.
            # This removes the magnitude mismatch between nats (logp) and [0,1] (LL).
            lps  = np.array([c["_norm_logp"] for c in candidates], dtype=float)
            lls  = np.array([c["ll_score"]   for c in candidates], dtype=float)
            lps_z = (lps - lps.mean()) / (lps.std() + 1e-8)
            lls_z = (lls - lls.mean()) / (lls.std() + 1e-8)
            for i, c in enumerate(candidates):
                score = (1.0 - blend_w) * lps_z[i] + blend_w * lls_z[i]
                if c["sentence_end"]:
                    score += lambda_fs * c["fs_score"]
                c["rerank_score"] = score
        else:
            # Additive mode (original).
            for c in candidates:
                score = c["_norm_logp"] + lambda_ll * c["ll_score"]
                if c["sentence_end"]:
                    score += lambda_fs * c["fs_score"]
                c["rerank_score"] = score

        # Sort by rerank then by logp for deterministic tie-breaking.
        candidates.sort(key=lambda c: (c["rerank_score"], c["_norm_logp"]), reverse=True)
        survivors = candidates[:beam_width]
        rejected  = candidates[beam_width:]

        # Count LL reorders: did rerank produce a different top-W than logp alone?
        logp_top_ids = sorted(range(len(candidates)), key=lambda i: candidates[i]["_norm_logp"], reverse=True)[:beam_width]
        rerank_top_ids = set(range(beam_width))
        if set(logp_top_ids) != rerank_top_ids:
            n_ll_reorders += 1

        if log_per_step:
            for c in survivors + rejected:
                step_log.append(StepRecord(
                    step=step,
                    parent_beam_id=c["parent_id"],
                    token_id=c["tok_id"],
                    token_text=c["new_token_text"],
                    token_logp=c["tok_logp"],
                    ll_score=c["ll_score"],
                    fs_score=c["fs_score"],
                    sentence_end=c["sentence_end"],
                    rerank_score=c["rerank_score"],
                    kept=c in survivors,
                ))

        # Build the new beam list from survivors.
        new_beams: list[_Beam] = []
        for c in survivors:
            parent = beams[c["parent_id"]]
            if parent.finished:
                # Carry the finished beam forward unchanged.
                new_beams.append(_Beam(
                    token_ids=list(parent.token_ids),
                    sum_logp=parent.sum_logp,
                    response_len=parent.response_len,
                    finished=True,
                    sentence_buffer=parent.sentence_buffer,
                    last_sentence_fs=parent.last_sentence_fs,
                    parent_id=c["parent_id"],
                ))
                continue
            new_beams.append(_Beam(
                token_ids=parent.token_ids + [c["tok_id"]],
                sum_logp=c["new_logp"],
                response_len=c["new_resp_len"],
                finished=bool(c["finishes"]),
                sentence_buffer=c["new_sentence_buffer"],
                last_sentence_fs=c["fs_score"] if c["sentence_end"] else parent.last_sentence_fs,
                parent_id=c["parent_id"],
            ))
        beams = new_beams

    # Final pick: highest length-normalised score among all surviving beams.
    def _final_score(b: _Beam) -> float:
        L = max(b.response_len, 1)
        return b.sum_logp / (L ** length_alpha) if length_alpha > 0 else b.sum_logp

    beams.sort(key=_final_score, reverse=True)
    best = beams[0]
    response_ids = best.token_ids[n_ctx:]
    text = tokenizer.decode(response_ids, skip_special_tokens=True)

    elapsed = time.time() - t0
    cfg_record = {
        "lambda_ll":         float(lambda_ll),
        "lambda_fs":         float(lambda_fs),
        "length_alpha":      float(length_alpha),
        "beam_width":        int(beam_width),
        "max_new_tokens":    int(max_new_tokens),
        "score_mode":        score_mode,
        "blend_w":           float(blend_w),
        "per_candidate_ll":  bool(per_candidate_ll),
        "seed":           int(seed),
        "fs_neutral":     float(fs_neutral),
    }

    return GuidedGenerationResult(
        text=text,
        response_token_ids=response_ids,
        final_score=float(_final_score(best)),
        n_steps=best.response_len,
        elapsed_s=float(elapsed),
        config=cfg_record,
        step_log=step_log,
        all_beams_text=[
            tokenizer.decode(b.token_ids[n_ctx:], skip_special_tokens=True) for b in beams
        ],
        n_ll_reorders=n_ll_reorders,
        n_sentence_end_steps=n_sentence_end_steps,
        n_total_steps=n_total_steps,
    )
