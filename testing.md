Assumptions

You use Sentence Transformers + Chroma (cosine, normalized).

You can export recent queries and the chunks/citations your system used.

1) Offline check (2–4 hours of work)

Goal: See if the gate beats the “≥0.8 verbatim” rule on your own data.

Dataset: 150–300 real queries covering FAQs, runbooks, error/lookups.

Labels (lightweight): For each query, a human marks: (a) correct/incorrect, (b) did the cited section actually contain the answer, (c) would verbatim be appropriate (FAQ/step).

Compare policies:

Baseline: “≥0.8 → verbatim; else compose.”

New gate: z-normalize + margin + coverage + consensus (+ exact-match).

Metrics to record:

Precision of final answers (target ≥90%).

False-verbatim rate (verbatim returned when it shouldn’t).

Deflection quality (“not confident” used correctly).

Top-5 evidence recall (should be ≥90%).

Decision rule: If precision improves ≥5 pts or false-verbatim drops ≥50%, proceed to shadow.

2) Shadow mode (no user impact)

Goal: Run the new gate in parallel, log decisions; users still see current outputs.

Duration: 3–7 days or ~1,000 queries.

Track per query:

Old vs new decision (verbatim/compose/deflect).

Normalized top1 score, margin, coverage, consensus flag.

Latency delta (should be ≈0; you’re not adding a reranker).

Spot-check 50–100 disagreements with quick human judgment.

Decision rule: If disagreements favor the new gate in ≥65% of spot-checks and latency impact <5%, go canary.

3) Canary rollout (small real traffic, easy rollback)

Goal: Prove in production on, say, 10–20% of traffic.

Guardrails:

Kill switch to revert instantly to the 0.8 rule.

Verbatim only when kind ∈ {FAQ, RunbookStep} (this single guard alone reduces risk a lot).

Exact-match required if the query includes an error code/log line.

Live metrics to watch (daily):

Acceptance rate (answers produced).

“Show me source” clicks and complaint/negative feedback rate.

Escalations/manual overrides from your team.

Latency p95.

Threshold calibration (one-time)

Start with coverage ≈ 0.40, margin Δ ≈ 0.35, z(top1) ≥ 0.

Sweep coverage 0.30→0.55 and margin 0.15→0.50 on the offline set.

Choose the pair that gives ≥90% precision with acceptable deflection.

Risks & how this plan de-risks them

“What if it blocks good answers?”
Canary + thresholds can be loosened; worst case flip the kill switch.

“What if it’s model-specific?”
You lock thresholds per embedding model; if you swap models, re-run the quick offline calibration.

“Latency?”
All signals are local math over the retrieved chunks—no reranker; p95 should be unchanged.

“Operator burden?”
Shadow first: you only review disagreements, not every query.

Success criteria (clear, binary)

Precision of final answers increases by ≥5 percentage points vs. 0.8 rule or false-verbatim halves.

Deflection (“not confident”) is used appropriately in ≥80% of reviewed deflections.

Latency p95 change < 5%.

User feedback (complaints/escalations) does not increase.

If it underperforms

Lower coverage by 0.05 or margin by 0.1 and re-check 50 queries.

If still weak, keep only the safest guards on top of your current rule:

Verbatim only for FAQ/RunbookStep.

Require exact match when queries contain error codes/log lines.

Cap verbatim to cases where two chunks from the same section agree.
