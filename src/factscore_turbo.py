"""
FActScore-Turbo: fast factual-precision scoring via a local LLM (Ollama).

Reference: https://habr.com/ru/companies/vk/articles/919852/

Pipeline
--------
1. Decompose the generated response → list of atomic facts.
2. Verify each fact against the source context (batch mode for speed).
3. Score = n_supported_facts / n_total_facts  ∈ [0, 1].

A low score (< threshold, default 0.5) signals a hallucinated response.
"""
from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

# ─── Prompts ──────────────────────────────────────────────────────────────────

_DECOMPOSE_SYSTEM = (
    "You are a text-analysis assistant. "
    "Your task is to decompose a response into atomic, self-contained factual claims. "
    "Each claim must be independently verifiable."
)

_DECOMPOSE_USER = """\
Decompose the following response into atomic facts.

Rules:
- Write ONE fact per line (no numbering, no bullets).
- Each fact must be self-contained — include the subject explicitly if needed.
- Omit filler phrases, greetings, hedges, and pure opinions with no verifiable content.
- If there are no verifiable facts in the text, output exactly: NO_FACTS

Response:
{response}

Atomic facts:"""

# ── Batch verification (one LLM call for all facts) ──────────────────────────

_BATCH_VERIFY_SYSTEM = (
    "You are a precise fact-checking assistant. "
    "Given a source context and a numbered list of claims, "
    "determine for each claim whether it is DIRECTLY supported by the context. "
    "Output ONLY a valid JSON array of booleans — no explanation, no extra text."
)

_BATCH_VERIFY_USER = """\
Source context:
\"\"\"
{context}
\"\"\"

Claims to verify:
{claims}

Output a JSON array with exactly {n} boolean values.
  true  → the claim is directly supported by the context above.
  false → the claim is not supported or contradicts the context.

JSON array:"""

# ── Single verification (fallback) ───────────────────────────────────────────

_SINGLE_VERIFY_SYS = "You are a precise fact checker. Answer with exactly one word."

_SINGLE_VERIFY_USER = """\
Source context:
\"\"\"
{context}
\"\"\"

Claim: {claim}

Is this claim directly supported by the source context?
Answer with exactly one word: SUPPORTED or NOT_SUPPORTED"""


# ─── Data classes ─────────────────────────────────────────────────────────────

@dataclass
class FActScoreResult:
    """Result of a single FActScore-Turbo evaluation."""

    score: float
    """Fraction of atomic facts supported by the source context ∈ [0, 1]."""

    facts: list[str] = field(default_factory=list)
    """Extracted atomic facts."""

    supported: list[bool] = field(default_factory=list)
    """Per-fact support flags (same order as `facts`)."""

    n_facts: int = 0
    n_supported: int = 0
    error: Optional[str] = None

    def is_hallucinated(self, threshold: float = 0.5) -> bool:
        """Returns True when score < threshold (response not grounded in context)."""
        return self.score < threshold

    def pretty_print(self) -> str:  # pragma: no cover
        lines = [f"FActScore: {self.score:.3f}  ({self.n_supported}/{self.n_facts} facts supported)"]
        for fact, sup in zip(self.facts, self.supported):
            lines.append(f"  {'✓' if sup else '✗'}  {fact}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"FActScoreResult(score={self.score:.3f}, "
            f"{self.n_supported}/{self.n_facts} supported)"
        )


# ─── Main class ───────────────────────────────────────────────────────────────

class FActScoreTurbo:
    """
    FActScore-Turbo evaluator backed by a local Ollama model.

    Parameters
    ----------
    model : str
        Ollama model tag.  Recommended for M1 Max 32 GB:
        - ``qwen2.5:7b``   (~5 GB, ~40 tok/s, best factual accuracy)   ← default
        - ``llama3.1:8b``  (~5 GB, strong instruction following)
        - ``qwen2.5:14b``  (~9 GB, higher quality, ~25 tok/s)
        - ``mistral:7b-instruct`` (~4.5 GB, fastest)
    temperature : float
        Sampling temperature (0 = greedy/deterministic, recommended).
    max_facts : int
        Maximum number of atomic facts to extract per response.
    batch_verify : bool
        Verify all facts in a single LLM call (much faster).
        Falls back to individual calls if JSON parsing fails.
    context_max_chars : int
        Context is truncated to this length before being sent to the LLM.
    max_retries : int
        Number of retries on Ollama API failures.
    """

    def __init__(
        self,
        model: str = "qwen2.5:7b",
        temperature: float = 0.0,
        max_facts: int = 15,
        batch_verify: bool = True,
        context_max_chars: int = 2_500,
        max_retries: int = 2,
    ) -> None:
        try:
            import ollama as _ollama
        except ImportError as exc:
            raise ImportError(
                "Ollama Python SDK not installed. Run: pip install ollama"
            ) from exc

        self._ollama = _ollama
        self.model = model
        self.temperature = temperature
        self.max_facts = max_facts
        self.batch_verify = batch_verify
        self.context_max_chars = context_max_chars
        self.max_retries = max_retries

    # ── Internal LLM helper ───────────────────────────────────────────────────

    def _chat(self, system: str, user: str, max_tokens: int = 512) -> str:
        messages = [
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ]
        for attempt in range(self.max_retries + 1):
            try:
                resp = self._ollama.chat(
                    model=self.model,
                    messages=messages,
                    options={"temperature": self.temperature, "num_predict": max_tokens},
                )
                return resp["message"]["content"].strip()
            except Exception as exc:
                if attempt >= self.max_retries:
                    raise RuntimeError(
                        f"Ollama call failed after {self.max_retries + 1} attempts: {exc}"
                    ) from exc
                wait = 1.5 ** attempt
                logger.warning("Attempt %d failed (%s) — retrying in %.1fs", attempt + 1, exc, wait)
                time.sleep(wait)
        return ""

    # ── Step 1: Decompose response → atomic facts ─────────────────────────────

    def decompose(self, response: str) -> list[str]:
        """Extract atomic facts from a generated response string."""
        text = response.strip()
        if len(text) < 15:
            return []

        raw = self._chat(
            _DECOMPOSE_SYSTEM,
            _DECOMPOSE_USER.format(response=text[:3_000]),
            max_tokens=600,
        )

        if "NO_FACTS" in raw.upper():
            return []

        facts: list[str] = []
        for line in raw.splitlines():
            # Strip bullets, dashes, numbering
            clean = re.sub(r"^[\s\-–•*]+\d*[\.\):]?\s*", "", line).strip()
            clean = re.sub(r"^\d+[\.\)]\s*", "", clean).strip()
            if len(clean) > 12:
                facts.append(clean)

        return facts[: self.max_facts]

    # ── Step 2: Verify facts against context ─────────────────────────────────

    def _verify_batch(self, facts: list[str], context: str) -> list[bool]:
        """Verify all facts with one LLM call (preferred path)."""
        ctx = context[: self.context_max_chars]
        numbered = "\n".join(f"{i + 1}. {f}" for i, f in enumerate(facts))
        raw = self._chat(
            _BATCH_VERIFY_SYSTEM,
            _BATCH_VERIFY_USER.format(context=ctx, claims=numbered, n=len(facts)),
            max_tokens=len(facts) * 12 + 30,
        )

        try:
            m = re.search(r"\[.*?\]", raw, re.DOTALL)
            if m:
                arr = json.loads(m.group())
                if len(arr) == len(facts):
                    return [bool(v) for v in arr]
        except (json.JSONDecodeError, ValueError):
            pass

        logger.warning("Batch JSON parse failed — falling back to single verification per fact")
        return [self._verify_single(f, context) for f in facts]

    def _verify_single(self, fact: str, context: str) -> bool:
        """Verify one fact (fallback)."""
        raw = self._chat(
            _SINGLE_VERIFY_SYS,
            _SINGLE_VERIFY_USER.format(
                context=context[: self.context_max_chars],
                claim=fact,
            ),
            max_tokens=10,
        ).upper()
        return "SUPPORTED" in raw and "NOT_SUPPORTED" not in raw

    # ── Public API ────────────────────────────────────────────────────────────

    def score(self, response: str, context: str) -> FActScoreResult:
        """
        Compute FActScore-Turbo for *response* grounded against *context*.

        Parameters
        ----------
        response : str
            The generated text to evaluate.
        context  : str
            The source passage(s) the response should be grounded in.

        Returns
        -------
        FActScoreResult
            score ∈ [0, 1] — fraction of atomic facts supported by context.
            score = 1.0 when no verifiable facts could be extracted.
        """
        try:
            facts = self.decompose(response)
            if not facts:
                return FActScoreResult(score=1.0, n_facts=0, n_supported=0)

            if self.batch_verify and len(facts) > 1:
                supported = self._verify_batch(facts, context)
            else:
                supported = [self._verify_single(f, context) for f in facts]

            n_sup = sum(supported)
            return FActScoreResult(
                score=n_sup / len(facts),
                facts=facts,
                supported=supported,
                n_facts=len(facts),
                n_supported=n_sup,
            )
        except Exception as exc:
            logger.error("FActScore computation error: %s", exc, exc_info=True)
            return FActScoreResult(score=float("nan"), error=str(exc))
