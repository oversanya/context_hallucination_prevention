"""
Internal helper for chat-style calls against a local Ollama server.

Centralises retry, timeout and option handling so that
:py:class:`FActScoreTurbo` and :py:class:`LLMJudge` share a single, well-tested
code path.
"""
from __future__ import annotations

import logging
import time
from typing import Optional

logger = logging.getLogger(__name__)


def ollama_chat(
    system: str,
    user: str,
    model: str,
    *,
    temperature: float = 0.0,
    max_tokens: int = 512,
    max_retries: int = 2,
    client=None,
) -> str:
    """
    Call ``ollama.chat`` with a (system, user) message pair.

    Parameters
    ----------
    system, user : str
        Message contents.
    model : str
        Ollama model tag (e.g. ``qwen2.5:14b``).
    temperature : float
        Sampling temperature.  0.0 is recommended for evaluation.
    max_tokens : int
        Upper bound on response length (``num_predict``).
    max_retries : int
        Number of additional attempts on Ollama failure (total = retries + 1).
    client : object, optional
        Ollama Python SDK module or compatible.  If None, imports lazily.

    Returns
    -------
    str
        The stripped assistant message content.

    Raises
    ------
    RuntimeError
        If all retry attempts fail.
    """
    if client is None:
        try:
            import ollama as client  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "Ollama Python SDK not installed. Run: pip install ollama"
            ) from exc

    messages = [
        {"role": "system", "content": system},
        {"role": "user",   "content": user},
    ]
    last_exc: Optional[Exception] = None
    for attempt in range(max_retries + 1):
        try:
            resp = client.chat(
                model=model,
                messages=messages,
                options={"temperature": temperature, "num_predict": max_tokens},
            )
            return resp["message"]["content"].strip()
        except Exception as exc:  # noqa: BLE001 — retry on any backend failure
            last_exc = exc
            if attempt >= max_retries:
                break
            wait = 1.5 ** attempt
            logger.warning(
                "Ollama attempt %d/%d failed (%s) — retrying in %.1fs",
                attempt + 1, max_retries + 1, exc, wait,
            )
            time.sleep(wait)
    raise RuntimeError(
        f"Ollama call to model {model!r} failed after {max_retries + 1} attempts: {last_exc}"
    )
