# Runbook — Guided Beam Search Milestone

End-to-end command sequence for launching the modified beam search experiment.
All paths are relative to the repository root.

---

## Prerequisites (one-time)

1. **Python deps** are already pinned in `requirements.txt` — verify with:
   ```bash
   uv pip install -r requirements.txt
   ```

2. **HF model** — Qwen-7B-Instruct (~14 GB float16):
   ```bash
   HF_HUB_ENABLE_HF_TRANSFER=1 python -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen2.5-7B-Instruct', allow_patterns=['*.json','*.txt','*.safetensors','*.tiktoken'])"
   ```

3. **Ollama models** — start the server, then pull both judges:
   ```bash
   ollama serve &
   ollama pull qwen2.5:14b   # ~9 GB, primary same-family judge
   ollama pull llama3.1:8b   # ~5 GB, cross-family judge (audit C1)
   ollama pull qwen2.5:7b    # ~5 GB — already present from FActScore work
   ```

4. **Disk space**: ~30 GB free recommended (models + checkpoints + parquet outputs).

---

## Step A — Retrain Lookback Lens classifier on Qwen-7B features (~30 min)

Required because Qwen-7B has 28 layers × 28 heads = 784 features vs 144 in the
opt-125m baseline.  The plan's audit-C3 acceptance is **test AUC ≥ 0.62 AND
train–test gap < 0.10**.

```bash
python experiments/lookback_lens_qwen/run.py
```

Outputs land in `experiments/lookback_lens_qwen/results/`:
- `classifier.pkl`, `metrics.json`, `features_full.npy`, `labels.npy`
- `roc_curve.png`, `coef_heatmap.png`

If acceptance fails, switch to PCA(k=64)+L2 by editing
`conf/lookback_lens_qwen.yaml` (`classifier.use_cv: false`,
`classifier.penalty: l2`) and rerun.

---

## Step B — λ-sweep on dev (~2–3 h)

Searches the (λ_LL, λ_FS) grid on n=20 dev contexts and picks the cell
maximising mean(faithfulness × completeness) across both judges.

```bash
python experiments/guided_generation/sweep.py
```

Outputs:
- `experiments/guided_generation/results/sweep_outputs.parquet` (resumable)
- `sweep_metrics.csv`     — per-cell aggregates
- `best_lambdas.json`     — chosen λs for the final run
- `splits.json`           — dev/test row identities (audit C2)

Wall-time budget: 4 cells × 3 cells × 20 rows × ~30 s/row ≈ 2 h with overhead.

---

## Step C — Final evaluation on test (~10–12 h overnight)

Five conditions × 200 contexts.  Per-row checkpointing every 10 rows so a
sleep / Ollama hiccup never costs more than 10 generations.

```bash
nohup python experiments/guided_generation/run.py > /tmp/guided_gen.log 2>&1 &
tail -f /tmp/guided_gen.log
```

Stages (in order):
1. Generation — fills `outputs.parquet` (~7 h).
2. Judge scoring (both judges, 3 dimensions) — `judge_scores.csv` (~1 h).
3. Pairwise A/B (3 method-vs-vanilla pairs × 2 judges) — `pairwise_ab.csv` (~1 h).
4. FActScore consistency check — `factscore_turbo.csv` (~1.5 h).

If interrupted, simply rerun: every stage uses (row_id, condition) keys for
resumability.

---

## Step D — Analysis (~1 min)

```bash
python experiments/guided_generation/analyse.py
```

Outputs:
- `analysis.json`           — paired bootstrap CIs, Wilcoxon p-values, judge κ,
                              length-collapse check, latency aggregates.
- `effect_sizes.png`        — bar chart of method effects vs vanilla_loop.
- `pareto.png`              — tokens/s vs faithfulness (audit R8).
- `latency_box.png`         — wall-time distribution per condition.
- `effect_sizes.csv`, `per_task_breakdown.csv`, `pairwise_summary.csv`.

---

## Step E — Examples notebook (~5 min interactive)

```bash
jupyter notebook notebooks/guided_generation_examples.ipynb
```

Runs all cells once.  After inspecting the auto-curated five examples,
optionally edit `CURATED_ROW_IDS` and re-run the rendering cell.  Exports the
blinded `human_spotcheck.csv` and `human_spotcheck_key.json`.

---

## Step F — Human spot-check (~30 min)

1. Read `experiments/guided_generation/results/human_spotcheck_instructions.md`.
2. Open `experiments/guided_generation/results/human_spotcheck.csv` in your
   editor and fill in `human_faithful`, `human_complete`, `human_coherent`.
3. Re-run the last cell of `guided_generation_examples.ipynb` to compute
   Cohen κ between your labels and the judge ensemble.

---

## Step G — Post-experiment audit

```bash
# Send analysis.json + the output artefacts back to experiment-auditor for
# audit_post.md (mandatory before the thesis section is written, per the plan
# Verification §8).
```

The auditor's report is saved to `experiments/guided_generation/audit_post.md`.

---

## Step H — Thesis section

Once the audit passes, write `report/sections/guided_beam_search.tex` using
the numbers from `analysis.json` and the qualitative examples in
`qualitative_examples.json`.  Final write-up cites:
- Lookback Lens (Chuang et al. 2024)
- Improved Beam Search (Marian et al. 2022) — see CLAUDE.md references
- The audit's flagged limitations (judge bias floor, n=200 power, etc.)
