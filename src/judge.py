"""
LLM-as-judge for the guided beam-search milestone.

Implements two evaluation modes:

1. **Likert scoring** — three independent 1-to-5 ratings per response on
   ``faithfulness``, ``completeness``, and ``coherence``, requested via a JSON
   prompt and parsed into a flat dict.  When several judge models are configured,
   each judge is queried independently and the mean / stddev across judges
   are reported alongside the per-judge values (audit C1: cross-family judges
   guard against same-family self-preference).

2. **Pairwise A/B preference** — given two responses for the same context and
   question, the judge picks A, B, or Tie.  Position bias is mitigated by an
   order-randomisation seed and the swap flag is logged so analysis can audit
   the randomisation.

Both modes share a strict JSON-only prompt and a forgiving regex fallback for
LLMs that wrap their JSON in markdown fences.
"""
from __future__ import annotations

import hashlib
import json
import logging
import random
import re
from dataclasses import dataclass
from typing import Iterable, Literal, Optional

import numpy as np

from ._ollama_chat import ollama_chat

logger = logging.getLogger(__name__)


# ─── Prompts ──────────────────────────────────────────────────────────────────

_LIKERT_SYSTEM = (
    "You are a careful, impartial evaluator. "
    "Score the assistant's response strictly against the provided context. "
    "Return ONLY a valid JSON object with three integer fields: "
    "faithfulness, completeness, coherence — each on a 1 to 5 scale. "
    "No prose, no markdown, no explanation."
)

_LIKERT_USER = """\
Context:
\"\"\"
{context}
\"\"\"

Question: {question}

Response:
\"\"\"
{response}
\"\"\"

Score the response on three independent dimensions:
- faithfulness (1=many claims contradict or are absent from context, 5=every claim is supported)
- completeness (1=ignores the question, 5=fully answers using context only)
- coherence (1=incoherent / ungrammatical, 5=clear, well-formed)

JSON:"""

_AB_SYSTEM = (
    "You are an impartial evaluator. "
    "Given a context, a question, and two candidate responses, "
    "decide which response is more faithful to the context "
    "and better answers the question. "
    "Reply with ONLY a valid JSON object {\"winner\": \"A\" | \"B\" | \"tie\"}. "
    "No prose, no markdown."
)

_AB_USER = """\
Context:
\"\"\"
{context}
\"\"\"

Question: {question}

Response A:
\"\"\"
{a}
\"\"\"

Response B:
\"\"\"
{b}
\"\"\"

Which response is more faithful and useful?
JSON:"""


# ─── Dataclasses ──────────────────────────────────────────────────────────────

@dataclass
class JudgeScore:
    """Aggregate Likert score for one (context, question, response) triple."""

    faithfulness: float
    completeness: float
    coherence:    float
    per_judge:    dict          # {model_tag: {dim: int}}
    judge_std:    dict          # {dim: float} stddev across judges, NaN if 1 judge
    parse_errors: list          # list of model tags whose output failed to parse

    def to_dict(self) -> dict:
        return {
            "faithfulness": self.faithfulness,
            "completeness": self.completeness,
            "coherence":    self.coherence,
            "per_judge":    self.per_judge,
            "judge_std":    self.judge_std,
            "parse_errors": self.parse_errors,
        }


@dataclass
class PairwiseResult:
    """Outcome of a single pairwise A/B preference call (one judge)."""

    judge_model:    str
    winner:         Literal["A", "B", "tie"]
    swap_flag:      bool        # True iff order was swapped before sending
    raw_response:   str
    parse_error:    bool


# ─── Judge ────────────────────────────────────────────────────────────────────

class LLMJudge:
    """
    Multi-model LLM-as-judge.

    Parameters
    ----------
    models : list of str
        Ollama model tags.  At least one is required.  Default uses two
        cross-family models (audit C1).
    temperature : float
        Sampling temperature for judge calls.  Default 0.0 (deterministic).
    max_tokens : int
        Maximum tokens per judge response.
    swap_seed : int
        Base seed for A/B order randomisation.  Per-call seeds are derived
        deterministically from (swap_seed, context, response_a, response_b).
    """

    def __init__(
        self,
        models: Optional[Iterable[str]] = None,
        *,
        temperature: float = 0.0,
        max_tokens: int = 80,
        swap_seed: int = 42,
        client=None,
    ) -> None:
        self.models = list(models) if models else ["qwen2.5:14b", "llama3.1:8b"]
        if not self.models:
            raise ValueError("LLMJudge requires at least one model tag.")
        self.temperature = temperature
        self.max_tokens  = max_tokens
        self.swap_seed   = swap_seed
        self._client     = client

    # ── Likert scoring ────────────────────────────────────────────────────────

    def score(self, context: str, question: str, response: str) -> JudgeScore:
        """Score one (context, question, response) on three Likert dimensions."""
        per_judge: dict[str, dict] = {}
        parse_errors: list[str] = []

        for model in self.models:
            raw = ollama_chat(
                _LIKERT_SYSTEM,
                _LIKERT_USER.format(context=context, question=question, response=response),
                model=model,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                client=self._client,
            )
            parsed = _parse_likert_json(raw)
            if parsed is None:
                logger.warning("Judge %s returned unparseable Likert: %r", model, raw[:200])
                parse_errors.append(model)
                continue
            per_judge[model] = parsed

        return self._aggregate(per_judge, parse_errors)

    @staticmethod
    def _aggregate(per_judge: dict[str, dict], parse_errors: list[str]) -> JudgeScore:
        dims = ("faithfulness", "completeness", "coherence")
        if not per_judge:
            nan = float("nan")
            return JudgeScore(
                faithfulness=nan, completeness=nan, coherence=nan,
                per_judge={}, judge_std={d: nan for d in dims},
                parse_errors=parse_errors,
            )
        means = {}
        stds  = {}
        for d in dims:
            vals = [float(s[d]) for s in per_judge.values() if d in s]
            means[d] = float(np.mean(vals)) if vals else float("nan")
            stds[d]  = float(np.std(vals, ddof=0)) if len(vals) >= 2 else float("nan")
        return JudgeScore(
            faithfulness=means["faithfulness"],
            completeness=means["completeness"],
            coherence=means["coherence"],
            per_judge=per_judge,
            judge_std=stds,
            parse_errors=parse_errors,
        )

    # ── Pairwise A/B ──────────────────────────────────────────────────────────

    def compare(
        self,
        context: str,
        question: str,
        response_a: str,
        response_b: str,
    ) -> list[PairwiseResult]:
        """
        Run a pairwise A/B preference call once per configured judge model.

        Returns
        -------
        list of :class:`PairwiseResult`, one per judge.  The ``swap_flag`` records
        whether (a, b) were submitted in the original order (False) or swapped
        (True); the returned ``winner`` is *always* in the original A/B frame
        (i.e. the de-swap is applied internally so callers never need to undo it).
        """
        # Deterministic per-row swap derived from (swap_seed, contents).
        swap_flag = _deterministic_swap(self.swap_seed, context, response_a, response_b)
        a_send, b_send = (response_b, response_a) if swap_flag else (response_a, response_b)

        results: list[PairwiseResult] = []
        for model in self.models:
            raw = ollama_chat(
                _AB_SYSTEM,
                _AB_USER.format(context=context, question=question, a=a_send, b=b_send),
                model=model,
                temperature=self.temperature,
                max_tokens=24,
                client=self._client,
            )
            winner = _parse_ab_json(raw)
            if winner is None:
                results.append(PairwiseResult(
                    judge_model=model, winner="tie", swap_flag=swap_flag,
                    raw_response=raw, parse_error=True,
                ))
                continue
            # De-swap so callers see results in the original A/B frame.
            if swap_flag and winner in ("A", "B"):
                winner = "B" if winner == "A" else "A"
            results.append(PairwiseResult(
                judge_model=model, winner=winner, swap_flag=swap_flag,
                raw_response=raw, parse_error=False,
            ))
        return results

    # ── Inter-judge agreement ─────────────────────────────────────────────────

    @staticmethod
    def cohen_kappa(labels_a: list, labels_b: list) -> float:
        """
        Cohen's kappa between two judges' integer-valued labels of the same length.

        Falls back to NaN when there is no observed disagreement variance.
        """
        from sklearn.metrics import cohen_kappa_score
        try:
            return float(cohen_kappa_score(labels_a, labels_b))
        except Exception as exc:  # noqa: BLE001
            logger.warning("Cohen kappa failed: %s", exc)
            return float("nan")


# ─── Parsing helpers ──────────────────────────────────────────────────────────

_INT_DIM_RE = re.compile(
    r"\"?(faithfulness|completeness|coherence)\"?\s*[:=]\s*([1-5])"
)


def _parse_likert_json(raw: str) -> Optional[dict]:
    """Parse three integer Likert scores from a judge response."""
    if not raw:
        return None
    # Strip markdown fences if present.
    cleaned = re.sub(r"```(?:json)?", "", raw).strip("` \n")
    # Try strict JSON first.
    try:
        m = re.search(r"\{.*?\}", cleaned, re.DOTALL)
        if m:
            obj = json.loads(m.group())
            scores = {k: int(obj[k]) for k in ("faithfulness", "completeness", "coherence") if k in obj}
            if len(scores) == 3 and all(1 <= v <= 5 for v in scores.values()):
                return scores
    except (json.JSONDecodeError, ValueError, TypeError):
        pass
    # Regex fallback for non-strict JSON outputs.
    matches = dict(_INT_DIM_RE.findall(cleaned))
    if len(matches) == 3:
        try:
            scores = {k: int(v) for k, v in matches.items()}
            if all(1 <= v <= 5 for v in scores.values()):
                return scores
        except ValueError:
            return None
    return None


_AB_RE = re.compile(r"\"?winner\"?\s*[:=]\s*\"?(A|B|tie)\"?", re.IGNORECASE)


def _parse_ab_json(raw: str) -> Optional[Literal["A", "B", "tie"]]:
    """Parse the winner field from a pairwise A/B judge response."""
    if not raw:
        return None
    cleaned = re.sub(r"```(?:json)?", "", raw).strip("` \n")
    try:
        m = re.search(r"\{.*?\}", cleaned, re.DOTALL)
        if m:
            obj = json.loads(m.group())
            w = str(obj.get("winner", "")).strip()
            if w.upper() in ("A", "B"):
                return w.upper()  # type: ignore
            if w.lower() == "tie":
                return "tie"
    except (json.JSONDecodeError, ValueError, TypeError):
        pass
    m = _AB_RE.search(cleaned)
    if m:
        token = m.group(1)
        if token.upper() in ("A", "B"):
            return token.upper()  # type: ignore
        return "tie"
    return None


def _deterministic_swap(swap_seed: int, *parts: str) -> bool:
    """Return a deterministic boolean swap flag derived from the inputs."""
    h = hashlib.sha256()
    h.update(str(swap_seed).encode("utf-8"))
    for p in parts:
        h.update(b"|")
        h.update(p.encode("utf-8", errors="replace"))
    # Use the first byte of the digest as an unbiased bit source.
    return bool(h.digest()[0] & 0x01)
