# Diagnostic Audit — Guided Beam Search "Null Result" Pilot

**Auditor:** experiment-auditor
**Date:** 2026-04-29
**Subject:** opt-125m pilot of `guided_beam_search` returning byte-identical text under all four conditions on 20/20 RAGTruth rows.
**Verdict:** **Diagnosis is MOSTLY CORRECT. Two important corrections below. Re-run with the proposed λ-sweep is NOT yet authorised — fix the design issue first.**

---

## Verdict on the diagnosis (per finding)

1. **Finding 1 (sdpa returns empty attentions; production code uses `attn_implementation="eager"`)** — **CORRECT.** Verified at `experiments/guided_generation/lib.py:95`. Your initial diagnostic was the buggy artefact, not the production loop. Good catch; document it explicitly so future auditors do not re-derive it.

2. **Finding 2 (LL is constant within parent beam)** — **CORRECT and the dominant root cause.** Confirmed by reading `src/guided_beam_search.py:252–260`: `ll_score` is computed **before** the loop over top-k candidates and is reused for all `beam_width` candidates from that parent. Therefore `λ_LL · LL` adds the **same constant** to every sibling and cannot reorder them. It can only reorder *across parents*. With `beam_width=2` (pilot config) there are only 2 parents, so LL has at most a 1-bit effect on rerank. This is a **design defect**, not a tuning issue.

3. **Finding 3 (cross-beam ΔLL ≈ 0.03–0.10 vs per-token Δlogp ≈ 1–5 nats)** — **CORRECT in direction, but the comparison is slightly mis-scaled.** With `length_alpha=0.7`, `norm_logp = sum_logp / L^0.7`, so the *marginal* change in `norm_logp` from adding one token at length L is roughly `Δlogp / L^0.7 − sum_logp · 0.7/L^1.7`. At L≈10–20, the effective scale of `norm_logp` differences is ~0.3–1.0 nats per candidate, not raw 1–5. Your conclusion still holds (LL≪logp), but the cleaner statement is: **the LL term and the length-normalised log-prob live on incommensurable scales, and the LL term has zero within-parent gradient** (Finding 2).

4. **Finding 4 (λ_LL=20 → degenerate output)** — **CORRECT.** This is the expected symptom of overwhelming logp with a constant that only orders parents: at high λ, the rerank picks whichever parent has the highest LL regardless of its log-prob, and the resulting beam is incoherent.

5. **Finding 5 (opt-125m peaked logits on RAGTruth)** — **CORRECT but partially confounded with Finding 2.** Even with non-peaked logits, Finding 2 alone forbids token-level differentiation. opt-125m's saturation makes the problem worse, not different.

---

## Methodological warnings

- **Do NOT claim in the thesis** that "Lookback-Lens guidance has no effect at the token level." The current implementation **cannot** have a token-level effect by construction. A negative claim requires a corrected implementation (per-candidate LL, see below).
- **Do NOT claim** the FS arm is null until you verify the `vanilla_loop` and `factscore` arms reach a sentence boundary within `max_new_tokens=48`. With opt-125m + beam_width=2, sentence-end tokens may never be selected within 48 steps; in that case `lambda_fs` is multiplied by zero firings and the null is mechanical.
- **Do NOT publish opt-125m as the headline model.** Document this as a code-correctness pilot only. The thesis claim must rest on a model whose attentions encode meaningful context-tracking (≥1B params, instruction-tuned).
- The `vanilla_hf` vs `vanilla_loop` byte-equivalence (per audit R3) is the *only* useful positive signal from this pilot — your custom loop is at least consistent with HF beam search at λ=0. Note that explicitly.

---

## Recommended remedy (numbered, in order)

1. **Fix the LL-is-constant defect (Finding 2).** Move the LL computation inside the candidate loop so each candidate's lookback ratio is computed on the partial response **including its own new token**. Concretely: append `tok_id` to `beam.token_ids`, run a second forward pass with `output_attentions=True`, extract features at `n_resp = beam.response_len + 1`, then compute `ll_score` per candidate. Cost: `beam_width × beam_width` extra forward passes per step instead of `beam_width`. For the pilot (bw=2, max_new=48) this is ~96 extra passes per row — trivial.
   - **Cheaper alternative if the cost matters at bw=4+ on Qwen-7B:** one forward over the parent beam *with the candidate token appended in a batched sequence dimension*, giving all beam_width LL values from a single forward by reading row i of the attention tensor. Document this optimisation only if you implement it.

2. **Fix the magnitude-mismatch (your Question 3 idea is correct).** Replace the additive form with a convex blend:
   `score = (1 − w) · z(norm_logp) + w · LL`, with `z(·)` a per-step standardisation (subtract step mean, divide by step std) over the candidate set, and `w ∈ {0, 0.1, 0.3, 0.5, 0.7}`. This makes `w` interpretable as the relative weight of the faithfulness signal and removes the need for ad-hoc λ values that depend on the model's logit scale. Apply the same scheme to FS at sentence boundaries.

3. **Add a sanity assertion that LL actually reorders candidates.** In `guided_beam_search`, when `lambda_ll > 0`, log the fraction of steps where the top-k candidates by `rerank_score` differ from the top-k by `norm_logp` alone. If this fraction is 0 in the pilot, the run is null *by construction* and must be flagged before any λ-sweep.

4. **Verify FS actually fires.** Add a counter `n_sentence_end_steps` to the result. If `lambda_fs > 0` but the counter is 0, the arm is mechanically null and the result is uninformative.

5. **Defer the λ-sweep until 1–4 are in.** A wider sweep on the *current* code will at best produce gibberish (high λ) or null (low λ) — both cases give nothing publishable. Only after Remedy 1 makes LL token-discriminative does sweeping λ make sense.

6. **Highest-leverage move under the bandwidth constraint:** I do **not** approve "rerun pilot with λ ∈ {0,5,10,15,20} on opt-125m" as proposed. Instead:
   - **Today:** apply Remedy 1 + 2 + 3 + 4 (a 1–2-hour code change). Re-run pilot at `w ∈ {0, 0.3, 0.5}` on opt-125m **only to validate that the corrected code produces non-identical outputs across conditions**. Acceptance criterion: at least one row in the pilot must show a different response under `lookback` vs `vanilla_loop`. If yes, the code is correct; if no, the diagnosis missed something else.
   - **In parallel:** switch the headline-model download from Qwen-7B-Instruct to a smaller eager-attention-capable model already viable on M1 Max — e.g. `Qwen2.5-1.5B-Instruct` (~3 GB) or `meta-llama/Llama-3.2-1B-Instruct` (~2.5 GB). These are large enough to have non-trivial attention structure, fast enough to iterate on, and avoid the 14 GB bandwidth block. Treat Qwen-7B as a stretch goal, not a precondition.
   - The Ollama-only models are unusable here — the LL signal is impossible without raw attention tensors. Do not pursue them.

7. **Do not authorise re-runs without explicit user permission.** Per project convention, the user must approve each re-run; this audit lists *what* to do, not *that* it has been authorised.

---

## Acceptance criteria for the next run

- ≥ 30% of pilot rows show non-identical outputs across at least two conditions.
- LL-reordering counter > 0 on a non-trivial fraction of steps.
- `n_sentence_end_steps` ≥ 1 on ≥ 50% of rows when FS is active.
- Identity preserved between `vanilla_hf` and `vanilla_loop` at `w=0` (regression check).
- All four diagnostic counters logged to parquet alongside `response`.

Only after these pass on the corrected pilot is a real `w`-sweep + headline-model run methodologically defensible.

---

*End of diagnostic audit.*
