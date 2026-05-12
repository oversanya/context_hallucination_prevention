# Human spot-check — instructions

Audit R6 of the guided-generation milestone.

## Purpose

You will rate 20 anonymised assistant responses on three independent
dimensions.  Your labels are compared with the LLM judges' Likert ratings to
estimate the bias floor of the automated evaluation.

The condition labels (vanilla / +LL / +FS / +LL+FS) are **hidden** behind
random IDs so your judgement is not biased by knowing which method generated
each response.  The mapping is stored separately and only revealed during
analysis.

## Workflow

1. Open `human_spotcheck.csv` (next to this file) in your editor of choice.
2. For each row, read **only** the `context`, `question`, and `response`
   columns.
3. Fill in three integer labels:
   - `human_faithful`  ∈ {0, 1}
   - `human_complete`  ∈ {0, 1}
   - `human_coherent`  ∈ {0, 1}
4. Save the CSV.
5. Re-run the last cell of `notebooks/guided_generation_examples.ipynb` to
   compute Cohen's κ between your labels and the judge ensemble.

Allow roughly 30–45 minutes for the full 20 rows.

## Label definitions

### `human_faithful` — "every claim is supported by the provided context"

- **1** — every factual claim in the response is *entailed* by something
  written in the context, paraphrasing allowed.  An "I cannot find this in the
  context" abstention is also faithful.
- **0** — at least one factual claim is *not* entailed by the context, or
  contradicts it, or adds external knowledge not present in the context.
  Speculation, fabricated names, fabricated numbers, or fabricated dates all
  count.

Rule of thumb: would a careful editor flag any sentence as "where does this
come from?" — if yes, label 0.

### `human_complete` — "the response answers the question using the context"

- **1** — the response addresses the question (or summarisation prompt) and
  includes the relevant information from the context.  An honest abstention
  ("not in the context") is acceptable when the context truly does not
  contain the answer; that should be labelled 1 because the response is the
  correct *complete* answer.
- **0** — the response ignores the question, gives a partial answer when the
  context contained more, or explicitly refuses when the answer was present.

### `human_coherent` — "the response is well-formed and readable"

- **1** — grammatical, on-topic, no repetition, no broken sentences.
- **0** — degraded fluency, mid-sentence cut-off, repeated phrases or tokens,
  contradictory clauses within the same response.

## Edge cases

- **Empty / very short response.** Faithful = 1 (no claims to falsify),
  complete = 0, coherent = 1 if grammatical.
- **Response just rephrases the context verbatim.** Faithful = 1,
  complete = 1 if it answers the question, coherent = 1.
- **Response is an honest abstention** ("Based on the provided context I
  cannot find …") when the context truly lacks the answer:
  faithful = 1, complete = 1, coherent = 1.

## Limitations to note in the thesis

This is a **single-annotator** spot-check.  Inter-annotator agreement is not
measured.  The result is reported as judge-vs-human Cohen κ on n=20 with this
limitation explicitly stated.
