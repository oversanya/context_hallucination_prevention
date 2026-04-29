"""
LookbackRatioExtractor: computes per-head lookback ratios from transformer attention.

The lookback ratio for layer l, head h at token position t is defined as:
    lookback_ratio[l, h, t] = sum(attn[l, h, t, context_positions])
                               / sum(attn[l, h, t, :])

A feature vector is the mean of these ratios over all response token positions,
yielding shape (n_layers * n_heads,).

Reference: Chuang et al. (2024), Section 3.
"""
from __future__ import annotations

import logging
import sys
import time
from typing import List

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)


class LookbackRatioExtractor:
    """
    Extracts lookback ratio feature vectors from a HuggingFace causal LM.

    Parameters
    ----------
    model_name : str
        HuggingFace model identifier.
    device : str
        Torch device string.  Use "mps" on Apple Silicon.
    max_context_tokens : int
        Context is truncated to this many tokens before concatenation.
        Setting a fixed budget guarantees response tokens always survive.
    max_response_tokens : int
        Response is truncated to this many tokens before concatenation.
    """

    def __init__(
        self,
        model_name: str = "facebook/opt-125m",
        device: str = "mps",
        max_context_tokens: int = 384,
        max_response_tokens: int = 128,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.max_context_tokens = max_context_tokens
        self.max_response_tokens = max_response_tokens

        logger.info("Loading tokenizer: %s", model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        logger.info("Loading model: %s on %s", model_name, device)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            output_attentions=True,
            torch_dtype=torch.float32,
        ).to(device)
        self.model.eval()

        # Derive architecture dimensions once.
        cfg = self.model.config
        self.n_layers: int = cfg.num_hidden_layers
        self.n_heads: int = cfg.num_attention_heads
        logger.info(
            "Model loaded — %d layers, %d heads, feature dim = %d",
            self.n_layers,
            self.n_heads,
            self.n_layers * self.n_heads,
        )

    def extract(self, context: str, response: str) -> np.ndarray:
        """
        Compute the lookback ratio feature vector for one (context, response) pair.

        Parameters
        ----------
        context : str
            Source passage the response should be grounded in.
        response : str
            Generated text to evaluate.

        Returns
        -------
        np.ndarray, shape (n_layers * n_heads,)
            Mean lookback ratio per (layer, head) over all response token positions.
            Returns a zero vector on error.
        """
        feature_dim = self.n_layers * self.n_heads

        # Tokenize context and response with independent budgets so that
        # long contexts cannot crowd out response tokens.
        ctx_enc = self.tokenizer(
            context,
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_context_tokens,
            return_tensors="pt",
        )
        resp_enc = self.tokenizer(
            response,
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_response_tokens,
            return_tensors="pt",
        )

        ctx_ids  = ctx_enc["input_ids"]   # (1, n_ctx)
        resp_ids = resp_enc["input_ids"]  # (1, n_resp)
        n_ctx  = ctx_ids.shape[1]
        n_resp = resp_ids.shape[1]

        if n_ctx == 0:
            logger.warning("Empty context after tokenization — returning zero vector.")
            return np.zeros(feature_dim, dtype=np.float32)
        if n_resp == 0:
            logger.warning("Empty response after tokenization — returning zero vector.")
            return np.zeros(feature_dim, dtype=np.float32)

        input_ids      = torch.cat([ctx_ids, resp_ids], dim=1).to(self.device)
        attention_mask = torch.ones(1, n_ctx + n_resp, dtype=torch.long).to(self.device)
        seq_len        = n_ctx + n_resp

        context_positions = list(range(n_ctx))

        # Forward pass — collect all attention tensors.
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_attentions=True,
            )

        # attentions: tuple of (1, n_heads, seq_len, seq_len) per layer.
        # Compute per-layer, per-head lookback ratios at response positions.
        response_positions = list(range(n_ctx, seq_len))
        features = np.zeros(feature_dim, dtype=np.float32)

        for l_idx, attn_layer in enumerate(outputs.attentions):
            # attn_layer: (1, n_heads, seq_len, seq_len)
            attn_np = attn_layer[0].cpu().float().numpy()  # (n_heads, seq_len, seq_len)
            for h_idx in range(self.n_heads):
                ratios = []
                for t in response_positions:
                    row = attn_np[h_idx, t, :]          # (seq_len,)
                    total = row.sum()
                    if total < 1e-12:
                        continue
                    ctx_mass = row[context_positions].sum()
                    ratios.append(ctx_mass / total)
                feat_idx = l_idx * self.n_heads + h_idx
                features[feat_idx] = float(np.mean(ratios)) if ratios else 0.0

        return features

    def extract_batch(
        self,
        contexts: List[str],
        responses: List[str],
        batch_size: int = 8,
    ) -> np.ndarray:
        """
        Extract lookback ratio feature vectors for a list of (context, response) pairs.

        Parameters
        ----------
        contexts : list of str
        responses : list of str
        batch_size : int
            Number of samples per logging checkpoint (actual computation is per-sample
            since sequence lengths vary).

        Returns
        -------
        np.ndarray, shape (n_samples, n_layers * n_heads)
        """
        n = len(contexts)
        assert len(responses) == n, "contexts and responses must have equal length."
        feature_dim = self.n_layers * self.n_heads
        X = np.zeros((n, feature_dim), dtype=np.float32)

        start_time = time.time()
        bar_fmt = "{l_bar}{bar}| {n}/{total} [{elapsed}<{remaining}, {rate_fmt}]"

        with tqdm(total=n, desc="Extracting lookback ratios", bar_format=bar_fmt, file=sys.stderr) as pbar:
            for i, (ctx, resp) in enumerate(zip(contexts, responses)):
                X[i] = self.extract(ctx, resp)
                pbar.update(1)

                # Periodic timing log and checkpoint.
                if (i + 1) % 25 == 0:
                    elapsed = time.time() - start_time
                    rate = (i + 1) / elapsed if elapsed > 0 else float("inf")
                    eta = (n - i - 1) / rate if rate > 0 else float("inf")
                    logger.info(
                        "Progress %d/%d — elapsed %.1fs — ETA %.1fs",
                        i + 1, n, elapsed, eta,
                    )

        return X
