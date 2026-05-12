# Pre-Implementation Audit — Guided Beam Search Milestone

**Auditor:** experiment-auditor (senior data scientist, scientific-rigor mode)
**Date:** 2026-04-29
**Subject of audit:** `/Users/vekshinkir/.claude/plans/ask-thesis-team-lead-to-implement-compressed-glade.md`
**Scope:** PLAN.md milestones §C (modified beam search) and §D (side-by-side examples).
**Verdict:** **GO-WITH-CAVEATS** — proceed with implementation only after the Critical issues below are resolved or explicitly accepted in writing.

---
v
## Executive summary

The plan is well-structured, reuses existing infrastructure, respects hardware constraints, and identifies the right scientific risks (self-judging, custom-loop bugs, λ leakage). However, several Critical methodological decisions are under-specified or actively dangerous as written:

1. **Same-family judge bias** is not quantified or mitigated.
2. **Dev/test split** has no implementation that guarantees disjointness; the existing `load_ragtruth()` API cannot produce two disjoint subsets in a single call.
3. **High-dimensional Lookback Lens classifier** (n=200 / p=784, after 80/20 split: n_train=160) is in n < p territory with default L2 `C=1.0`, no regularisation tuning, no PCA/feature selection.
4. **Length-normalisation** of beam scores is not specified — short responses can trivially game FActScore (zero atomic facts → score=1.0).
5. **Statistical power** at n=100 is borderline for the small-to-medium effects expected here.

None of these is fatal. All are fixable with bounded effort *before* the 6–8 h overnight run.

---

## Verdict and decision

> **GO-WITH-CAVEATS.** Address Critical issues 1–5 before writing any production code. Recommended improvements are non-blocking but should be incorporated where they cost little. Re-audit is not required after the fixes — fixes should be self-evidenced by config diffs and a short addendum at the top of `audit_pre.md`. A **post-hoc audit** is mandatory before the thesis section is finalised (per the plan's own `audit_post.md` requirement).

---

## Critical issues (must fix before any code)

### C1. Same-family judge bias is not quantified.
- **Severity:** Critical.
- **Explanation:** Generator = Qwen2.5-7B-Instruct (HF). Primary judge = qwen2.5:14b (Ollama). Same vendor, same instruction-tuning lineage, same tokenizer family, very likely overlapping pre-training distributions. Literature on LLM-as-judge (Zheng et al. 2023 — MT-Bench, Panickssery et al. 2024 — "LLM Evaluators Recognize and Favor Their Own Generations") shows self-preference bias on the order of 5–25 percentage points on pairwise A/B and ~0.2–0.5 Likert points on faithfulness for same-family pairs. This is *larger* than the effect size you are likely to detect from guided beam search. A positive judge result with this setup is not interpretable as a real faithfulness gain — it could be entirely explained by family preference if the +LL+FS condition produces text more stylistically similar to Qwen-14B's preferred phrasing (which is highly plausible because LL signal pulls toward context tokens, and FS signal favours decomposable factual claims — both of which qwen2.5:14b is itself optimised to emit).
- **Required fix:** Add a **second judge from a different model family** for the *primary* evaluation, on the *full* test set (not just a subset). Recommended: `llama3.1:8b` via Ollama (fits comfortably on M1 Max, ~5 GB, fast). The primary headline number must be "agreement between two cross-family judges" or the *average* of the two. If the user wants to keep qwen2.5:14b as the main judge for cost reasons, they must additionally include llama3.1:8b on at least 50% of the test set as a sensitivity analysis, and report **judge–judge κ** so the reader can see the bias floor. A GPT-4o subset (n=20) is nice-to-have but is not a substitute — it is a third-party check, not the primary judge.
- **Why this matters for the thesis defence:** A reviewer will ask "could the improvement be self-preference bias?" within the first three minutes of Q&A. Without a cross-family judge you cannot answer that question.

### C2. Dev/test split is not provably disjoint.
- **Severity:** Critical.
- **Explanation:** `src/benchmark.py:load_ragtruth(n_samples, seed)` performs *stratified balanced sampling*: it splits into pos/neg, takes `n_each = min(n_samples//2, len(pos), len(neg))` from each, concatenates and shuffles. There is **no API to request two disjoint subsets**. Two natural calls — `load_ragtruth(20, seed=42)` for dev and `load_ragtruth(100, seed=42)` for test — share rows whenever the underlying dataframe ordering is stable, because both calls draw from the same `pos.sample(..., random_state=42)` start. Worse: even with different seeds, `pos.sample(20, seed=A)` and `pos.sample(50, seed=B)` will overlap on roughly `20·50/|pos|` rows by birthday-paradox math — non-zero leakage. The plan does not specify how disjointness is enforced.
- **Required fix:** Either (a) add a `load_ragtruth_split(n_dev, n_test, seed)` helper that draws `n_dev + n_test` balanced rows once and returns two disjoint frames; OR (b) in the experiment script, call `load_ragtruth(n_samples=120, seed=42)` and slice `df.iloc[:20]` for dev and `df.iloc[20:120]` for test. Document the chosen approach in the YAML config. Add a `pytest` assertion: `assert set(dev.index) & set(test.index) == set()`. Save the exact row-id list for both splits to `experiments/guided_generation/results/splits.json` so reviewers can reproduce.
- **Bonus risk this fixes:** With n=20 dev + n=100 test = 120 total balanced rows, you also need to verify RAGTruth has ≥60 hallucinated and ≥60 faithful rows that survive the length filters (`len > 30`, `len > 10`). If `n_each = min(60, len(pos), len(neg))` clips below 60, your split silently shrinks.

### C3. n_train=160, p=784 — high-dimensional regime with no regularisation tuning.
- **Severity:** Critical.
- **Explanation:** Confirmed via `src/lookback_lens/classifier.py:41` — `LogisticRegression(max_iter=…, solver="lbfgs", C=1.0)` with default L2 and no cross-validation. Qwen2.5-7B has 28 layers × 28 heads = 784 features; the existing baseline (opt-125m, p=144) had n/p ≈ 1.1 (already marginal) and reached AUC=0.625. With p=784 and n_train≈160, n/p ≈ 0.20 — you are squarely in the **n < p high-dimensional regime**. With L2 `C=1.0` (mild penalty) and no scaling check, you are highly likely to overfit on training and report a misleadingly low test AUC (or the opposite — get a lucky high AUC that does not generalise).
- **Required fix:** Three changes, in order of priority:
  1. **Use `LogisticRegressionCV`** with internal stratified 5-fold CV to pick `C` from a log grid `[1e-3, 1e-2, 0.1, 1, 10]`. Cost: ~seconds. This is a one-line change in `LookbackLensClassifier.__init__` and is good hygiene for the existing baseline too.
  2. **Strongly consider L1 (`penalty="l1", solver="liblinear"` or `saga`)** to get sparsity. Lookback-lens features are known to be redundant — many heads carry no signal. L1 will pick a subset.
  3. **Report n_train, n_test, n_features, chosen C, n_nonzero coefficients** in `metrics.json`. The acceptance criterion `AUC ≥ 0.60` is too lenient given the prior baseline already hit 0.625 — set it at **AUC ≥ 0.62** at minimum, and require the train–test gap `< 0.10` to flag overfitting.
- **PCA alternative:** Optional. A PCA to `k=64` components before logistic regression is a defensible alternative if L1 does not converge well; mention in the report which one was chosen and why.

### C4. Length-normalisation of beam scores is unspecified.
- **Severity:** Critical.
- **Explanation:** The scoring formula `score = logp + λ_LL·LL + 𝟙[sentence_end]·λ_FS·FS` adds *cumulative* `logp` (negative, scales with length) to per-step or per-sentence bonuses (bounded). This will systematically prefer **shorter responses** in the +LL+FS arm because the FS bonus only fires at sentence boundaries. Worse, FActScore-Turbo's score is *defined as* `n_supported / n_facts` — when `n_facts == 0` (response too short to extract any claim), most implementations either return 1.0 (perfect score, tested in your `factscore_turbo.py`) or NaN. Either way, "say less to score better" becomes a degenerate dominant strategy. This is a textbook beam-search failure mode and the very issue the *Improved Beam Search for Hallucination Mitigation* paper (which CLAUDE.md cites) addresses.
- **Required fix:**
  1. **Length-normalise `logp`** by sequence length: use `logp / (length ** α)` with `α ∈ [0.6, 1.0]` (Wu et al. 2016 GNMT formulation). Default `α=0.7`.
  2. **Cap response length floor** in the FS scorer or in the rerank: if `n_facts == 0`, return a *neutral* score (0.5), not 1.0, so empty/degenerate beams are not rewarded.
  3. Add a **sanity assertion** in the analysis step: flag any condition whose median response length is < 0.7× the vanilla median — that is the canonical signature of length collapse.
  4. **Verify** in `tests/test_guided_beam_search.py` that with `λ_LL = λ_FS = 0` the output exactly matches a length-normalised vanilla beam search at the same `α`.

### C5. Statistical power at n=100 is borderline for the expected effect.
- **Severity:** Critical (becomes Major if the user explicitly accepts the lower power).
- **Explanation:** For a paired test (Wilcoxon signed-rank, two-sided, α=0.05) to detect Cohen's d=0.3 (small-to-medium) at 80% power, you need approximately **n=85 paired samples** (Wilcoxon's asymptotic relative efficiency to t-test is ~0.955; the t-test requirement is n≈82). At n=100 you have ~85% power for d=0.3 and ~99% for d=0.5. This is *fine* if the true effect is d ≥ 0.3, **but inadequate if d ≈ 0.2** — which is the realistic effect size for guidance signals on a strong base model like Qwen-7B. Lookback Lens at offline AUC=0.625 is a weak signal; integrating it into beam search typically yields d=0.1–0.25 in published results.
- **Required fix:** Either (a) increase final test set to **n=200** (doubles wall-time to 12–16 h — split across two nights, with checkpointing), giving 80% power for d≈0.20; OR (b) explicitly state in the report the **minimum detectable effect** at n=100 (d≈0.28 for 80% power), and pre-commit that effects below that will be reported as "underpowered, no claim of improvement." Option (b) is cheaper but commits you to honest negative reporting. Pick one before you start.
- **Implementation note:** Paired bootstrap with B=10 000 (already in plan) gives valid CIs at any n; the issue is *power*, not validity. CI width at n=100 for a faithfulness difference is roughly ±0.15 Likert points — wider than the effect size you are trying to detect.

---

## Recommended improvements (should incorporate; not blocking)

### R1. FActScore-Turbo as evaluation column → label as "consistency check," do not headline.
The plan correctly uses qwen2.5:14b as judge and qwen2.5:7b as both guidance signal and final FActScore-Turbo evaluator. The plan even acknowledges this in Step 3. The +FS guided arm is structurally biased on the FActScore-Turbo evaluation column (it was *optimised for it*). **Recommendation:** keep FActScore-Turbo in the results table but mark it explicitly with an asterisk: "* sanity / consistency check; gameable by +FS arm; not used to claim improvement." The headline numbers must come from (a) the cross-family judge (see C1) and (b) the human spot-check (see R6). Do not drop the FActScore column — it is useful as a sanity check that the FS signal is doing what you trained it to do.

### R2. Pairwise A/B should cover all three method-vs-vanilla comparisons.
Currently only `vanilla vs +LL+FS` is in the pairwise pass. Ablations matter: if `+LL+FS` wins but `+LL` alone and `+FS` alone do not, then the combination is what works. Add `vanilla vs +LL` and `vanilla vs +FS` to the pairwise judge — same n=100, same order-randomisation, same judge. Cost: ~2× the pairwise wall-time (still small relative to generation). This is required to support any claim that "both signals are necessary."

### R3. Add an HF `model.generate` reference baseline as a sanity check.
Risk: a custom beam-search loop has subtle bugs that can corrupt the "vanilla" arm (`λ=0`). Recommended: include a **fifth condition** `vanilla_hf` = `model.generate(num_beams=4, do_sample=False, length_penalty=0.7, max_new_tokens=128, ...)` from HuggingFace, run on the same n=100 test set with the same prompt format. Compare `vanilla_hf` against custom `vanilla_loop` (`λ_LL = λ_FS = 0`) on **token-level identity** for a small subset (n=5) and on **judge faithfulness distribution** for the full set. If they differ materially, your custom loop has a bug and *all your guided arms are also bugged*. This is a 1 hr cost that bullet-proofs the entire experiment.

### R4. Greedy and nucleus baselines for completeness.
For the thesis defence, a reviewer will ask "is beam search even the right starting point?" Add two cheap baselines on the same n=100:
- `greedy` (HF `do_sample=False, num_beams=1`)
- `nucleus` (`top_p=0.9, temperature=0.7, seed=42, num_beams=1`)
These cost almost nothing (no FActScore calls during decoding). They contextualise the +LL+FS gain: if guided beam search is just slightly above greedy, the contribution is weaker than if it dominates both.

### R5. Wall-time and resumability — verify checkpoint pattern is reused.
Confirmed: `experiments/lookback_lens_baseline/run.py:_extract_with_checkpoints` is a working pattern. **Required for the guided generation run:** save `outputs.parquet` incrementally every 10 generations (not at the end). The 6–8 h run will be interrupted if the laptop sleeps, the user accidentally closes the lid, or Ollama hiccups. Without per-row checkpointing, you lose the entire run. The plan says "reuse the checkpoint pattern" but does not name the file/granularity — make it explicit in `run.py`.

### R6. Human spot-check methodology — minimum acceptable bar.
A 20-row CSV with a blank `human_label` column is **insufficient** as written. For a Master's thesis, the minimum acceptable methodology is:
1. **Written instructions** (1 page) defining "faithful" precisely (e.g., "every claim is entailed by the provided context; no claim contradicts the context; no claim adds external knowledge"). Save as `human_spotcheck_instructions.md`.
2. **Two annotators** if at all feasible (the user themselves + one peer/advisor). Compute Cohen's κ. Single-annotator scoring is acceptable but must be flagged as a limitation.
3. **Blinded condition assignment** in the CSV — annotator must not know whether a row is `vanilla` or `+LL+FS`. Shuffle and assign random IDs; map back after annotation.
4. **Three labels per row, not one:** `faithful (0/1)`, `complete (0/1)`, `coherent (0/1)`. Even single-annotator multi-dimensional scoring is more informative than a single bool.
5. **Inter-annotator agreement *and* judge-vs-human agreement** computed in the notebook re-run cell.

This is not optional for a thesis — examiners will not accept a single blank-CSV column as "human evaluation."

### R7. Log per-condition decoding determinism.
The custom loop must fix all RNG state explicitly: `torch.manual_seed(seed)`, `torch.mps.manual_seed(seed)` (MPS-specific), `numpy.random.seed(seed)`, and `random.seed(seed)` at the start of each generation. Beam search with `do_sample=False` should be deterministic anyway, but MPS has known non-determinism in attention kernels (PyTorch issues #77764, #87657). Add an integration test: run the same context twice and assert identical outputs. If MPS introduces non-determinism, fall back to CPU for the analysis subset (n=20 examples for the §D notebook), or document the noise floor.

### R8. Report tokens/sec and memory per condition, not just wall-time.
The plan mentions latency tables. Make them concrete: for each condition, report `mean tokens/s`, `median wall-time per response`, `peak GPU memory (MPS)`, and `total Ollama overhead`. Plot a Pareto curve: x=tokens/s, y=judge-faithfulness mean. This is what allows a reviewer to assess "is the complexity worth it?" — required by CLAUDE.md.

---

## Non-issues (considered and dismissed)

- **Reuse of the FActScore-Turbo Ollama wrapper for the judge** — fine; extracting `_chat()` to a shared module is good hygiene and does not affect scientific validity.
- **Beam width = 4** — sensible default for M1 Max; not a confounder as long as it is identical across all conditions (verify in code).
- **Sentence-boundary FS triggering only** — correct trade-off between cost and signal density; reviewers will accept this.
- **Hydra/OmegaConf config logging** — already established convention, plan respects it.
- **Custom beam-search loop in ~250 lines** — pedagogically defensible for a thesis; the risk is bugs (addressed in R3), not the choice itself.
- **n=20 dev for sweep** — fine *for ranking λ values*, not for absolute claims. Plan uses it correctly (best-cell pick, not effect-size estimation).
- **Float16 on MPS for Qwen-7B** — standard; ~14 GB fits the budget. Not an audit concern.
- **Caching/reuse of attentions from the generator forward pass** — correct optimisation, no extra forward pass needed.
- **Curated examples (n=5)** as illustrations — fine; they are illustrative, not evidentiary, and the plan correctly does not draw quantitative claims from them.

---

## Required fixes summary (checklist)

Before writing any code:

- [ ] **C1** — Add cross-family judge (`llama3.1:8b` Ollama). Make it primary or report κ.
- [ ] **C2** — Implement disjoint-split logic; assert no row overlap; persist `splits.json`.
- [ ] **C3** — Switch to `LogisticRegressionCV` (or L1); raise AUC acceptance to ≥0.62 with train–test gap <0.10.
- [ ] **C4** — Length-normalise `logp` (α=0.7); neutralise FS=0.5 when n_facts=0; assertion on length collapse; vanilla-equivalence test.
- [ ] **C5** — Either bump n_test to 200 OR pre-commit minimum detectable effect (d≈0.28) and honest negative reporting.
- [ ] **R2** — Pairwise A/B for `vanilla vs +LL` and `vanilla vs +FS` in addition to `vanilla vs +LL+FS`.
- [ ] **R3** — Add `vanilla_hf` HF reference condition for sanity.
- [ ] **R6** — Human spot-check needs written rubric, blinded assignment, and 3 dimensions (not one blank column).

Recommended (non-blocking but valuable):

- [ ] **R1** — Label FActScore evaluation column as consistency check, not improvement evidence.
- [ ] **R4** — Add greedy and nucleus baselines.
- [ ] **R5** — Per-row incremental checkpointing of `outputs.parquet`.
- [ ] **R7** — Determinism-logging and MPS non-determinism check.
- [ ] **R8** — Pareto curve (cost vs gain) in the analysis output.

---

## Re-run authorisation

This is a pre-experiment audit. **No re-run authorisation is being granted here** — no experiment has been run yet for this milestone. Implementation may proceed only after Critical issues are resolved. The post-hoc audit (`audit_post.md`, per the plan's own §Verification line 8) remains mandatory before the thesis section is written.

---

*End of audit.*
