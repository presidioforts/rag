# A+ Retrieval Without Generation for DevOps: A Multi-Signal Gate Over Sentence-Transformer + Chroma

## Abstract

We present a **no-LLM**, low-latency retrieval system that delivers **A+ quality** answers for DevOps support by combining (1) a fine-tuned **SentenceTransformer** retriever over Confluence content, (2) two complementary indices—**FAQ** (Q→A pairs) and **Chunks** (runbooks & troubleshooting spans), and (3) a **multi-signal gating policy** that replaces brittle cosine cutoffs with robust decision logic (z-score margin, IDF-coverage, section-level consensus, and literal exact-match). The design returns **verbatim** answers only when safe, otherwise **deflects** with suggested sections, maintaining high precision without any generation.

---

## 1. Problem & Requirements

**Problem.** Naive semantic similarity (e.g., cosine ≥ 0.8) often yields **false positives**, especially on near-duplicate sections and short queries, and lacks safeguards for error-code/log queries.

**Non-Goals.** No LLMs, no generative summaries. Answers must be **verbatim** from vetted content (FAQ or documentation chunks).

**Targets.**

* End-to-end precision (human-judged) ≥ **90%**
* Chunk retriever Recall\@5 ≥ **90%** on held-out queries
* Low latency with small top-k (FAQ 10, Chunks 20)

---

## 2. Corpus Surfaces (Three Designs)

### 2.1 FAQ (Short, Stable Answers)

* **Format (your schema):**

  ```json
  {"input":"How do I create a Markdown table?",
   "target":"Use pipe tables with a dashed header row.\n\n| H1 | H2 |\n|----|----|\n| A  | B  |",
   "meta":{"page_title":"Markdown Style Guide","url":"https://…/markdown#tables","section_path":"markdown/tables","updated_at":"2025-07-01","kind":"FAQ","topic":"markdown"}}
  ```
* **Embedding field:** `input` (question).
* **Returned field:** `target` (verbatim answer).
* **Index:** `faq_collection` (Chroma).
* **Use:** Known questions; short answers; policy/definition.

### 2.2 Runbook (Procedures/Steps)

* **Format (schema):**

  ```json
  {"input":"CIWAT-SCP service restart (runbook step)",
   "target":"Service restart\n1) SSH to host ciwat-01\n2) sudo systemctl restart scp-ciwat.service\n3) sudo systemctl status scp-ciwat.service",
   "meta":{"page_title":"CIWAT-SCP Runbook","url":"https://…/runbook#service-restart","section_path":"runbook/service-restart","updated_at":"2025-06-12","kind":"RunbookStep","component":"ciwat-scp"}}
  ```
* **Embedding field:** `target` (chunk text).
* **Returned field:** `target` (verbatim step(s)).
* **Index:** `chunk_collection`.
* **Use:** “how/steps/restart/deploy/rollback”.

### 2.3 Troubleshooting (Errors/Logs/Incidents)

* **Format (schema):**

  ```json
  {"input":"K8s OOMKilled troubleshooting",
   "target":"Troubleshooting OOMKilled\nSymptoms: Pod terminated with OOMKilled.\nFix:\n- Delete failed pod\n- Increase memory requests/limits\nValidation: kubectl get pods shows a new Running pod",
   "meta":{"page_title":"Kubernetes Troubleshooting","url":"https://…/k8s/troubleshooting#oomkilled","section_path":"k8s/troubleshooting/oomkilled","updated_at":"2025-05-20","kind":"Troubleshooting","component":"kubernetes"}}
  ```
* **Embedding field:** `target`.
* **Use:** Error codes, exceptions, stack traces, quoted log lines.

---

## 3. Why ≥0.8 Cosine Is Not Enough

Cosine alone does not normalize **per-query score distributions**, cannot enforce **uniqueness** or **freshness**, ignores **lexical coverage**, and fails on **literals** (e.g., `"OOMKilled"`, `"HTTP 503"`). The result: false-verbatim answers and brittle behavior across topics.

---

## 4. Multi-Signal Gate (Decision Policy)

### 4.1 Signals

Given a query *q* and top-N candidates with raw similarity scores *s₁..sN*:

1. **Per-query normalization:**
   $z_i = \frac{s_i - \mu}{\sigma}$ over the candidate set.

2. **Margin test (disambiguation):**
   $z_1 - \max(z_2..z_5) \ge \Delta$ (start with $\Delta = 0.35$).

3. **IDF-weighted coverage (faithfulness):**
   Let $Q$ be tokens of *q*, $U$ union of tokens from top-3 chunks, and IDF from $\{Q\} \cup \text{top-3}$.
   $\text{coverage} = \frac{\sum_{t \in Q \cap U} \text{IDF}(t)}{\sum_{t \in Q} \text{IDF}(t)}$
   Threshold: **0.40**.

4. **Consensus (redundancy from same section):**
   Group by `(url, section_path)`. Support a group if a chunk contains at least \~⅓ of salient query terms (min 1). Require **≥2 chunks** supported for pass (FAQ often exempt; Runbook/Troubleshooting prefer it).

5. **Exact-match literals (safety):**
   If *q* contains error-ish tokens or quoted strings, require **at least one** top chunk to contain **all** literals.

6. **Freshness prior (tie-breaker):**
   Multiply by $w_t = \exp(-\lambda \cdot \Delta\text{days})$ (e.g., half-life ≈ 365 days).

7. **Verbatim eligibility:**
   Allow verbatim only for `kind ∈ {FAQ, RunbookStep}` and **Jaccard(query, text) ≥ 0.60**.

### 4.2 Return Policy

* **VERBATIM**: Gate passes; kind eligible; lexical overlap high → return `target` exactly.
* **COMPOSE**: Gate passes but not verbatim-eligible → return a compact **stack** of corroborated snippets (still verbatim text, no rephrasing).
* **DEFLECT**: Gate fails → return up to 3 suggested sections + one clarifying question.

---

## 5. Inference Pipeline (No LLM)

1. **Embed** query once (SentenceTransformer).
2. **Parallel search**:

   * FAQ: top-k = **10** (early-exit allowed).
   * Chunks: top-k = **20** (per-section cap = 3).
3. **Router heuristic**:

   * If error/log literals → **Troubleshooting (Chunks)**.
   * Else if procedural verbs (“how/steps/restart/deploy/rollback”) → **Runbook (Chunks)**.
   * Else → **FAQ first**, fallback to Chunks on gate fail.
4. **Apply gate** on chosen surface.
5. **Answer**:

   * VERBATIM: return `target` + citation (page\_title, section\_path, url).
   * COMPOSE: return 2–3 corroborated snippets (verbatim) + citations.
   * DEFLECT: suggestions + clarifying question (no generation).

**Top-k tuning:** Start with FAQ=10 and Chunks=20; add **early-exit** when margin+coverage pass after top-10 to reduce latency.

---

## 6. Dataset Preparation

### 6.1 FAQ (Q→A)

* Each item as per your schema.
* 3–5 **paraphrases** per question pointing to the same `target`.
* Keep answers ≤120 words; atomic; link back to source via `meta.url`.

### 6.2 Chunking (Docs)

* 200–400 tokens; split at headings; 10–15% overlap.
* Preserve code blocks; flatten small tables if needed (“key: value”).
* Metadata: `page_title, url, section_path, updated_at, kind, component, env_scope`.

### 6.3 Retriever Training Triples (for Chunks)

* **Positive**: the exact chunk containing the answer.
* **Hard negatives (3–5)**: near-miss chunks (esp. same/adjacent sections).
* Use **MultipleNegativesRankingLoss** (InfoNCE).
* **Two-round mining**: train → re-embed → mine new confusers → retrain (1 epoch).

---

## 7. Training Strategy (SentenceTransformer)

**Backbone:** `all-mpnet-base-v2` or `all-MiniLM-L6-v2` (384-d) for speed.
**Loss:** MultipleNegativesRankingLoss.
**Hyper-params:** lr 2e-5; batch 32–64; 1–3 epochs; cosine (L2-normalized).
**Optional:** TSDAE pre-adaptation on unlabeled Confluence text.
**Acceptance:** Recall\@5 ≥ **90%** on a 150–300 query hold-out; stable z-margin distributions.

---

## 8. Evaluation & Calibration

### 8.1 Metrics

* **Retriever:** Recall\@k (k=5,10), MRR\@10.
* **End-to-end:** human-judged precision (≥90%), deflect rate, false-verbatim rate.
* **Routing:** % FAQ vs Chunks; literal-satisfaction rate on error queries (≥95%).

### 8.2 Calibration Procedure

1. Fix chunking rules & per-section cap.
2. Grid-search **k ∈ {8,12,20,30}** → choose smallest k within 1–2 pts of best Recall\@5.
3. Tune **MARGIN\_DELTA** (0.25–0.45) and **COVERAGE\_MIN** (0.30–0.50) to keep precision ≥90%.
4. Enable freshness prior; audit old pages dominating top-k.

### 8.3 Ablations (suggested)

* With/without **coverage**; with/without **consensus**; with/without **exact literals**; different **k** settings; with/without **freshness**.

---

## 9. Observability & Governance

**Per-query logs:** `index_used (faq|chunk)`, `z1`, `margin`, `coverage`, `consensus`, `literals`, `literals_satisfied`, `citations`, `updated_at`, `per_section_cap_hit`, `decision`, `deflect_reason`.

**Dashboards:**

* Gate pass/fail breakdown; reasons over time.
* Drift: flat z-score distributions; increased deflects due to coverage.
* Freshness skew: old content over-selected.

**Safety & Ops:**

* Always **verbatim**; no rewriting.
* Command answers include a short safety footer (optional): “Verify in non-prod first.”
* DEFLECT on ambiguous acronyms or missing literals.

---

## 10. Production Defaults (Good Starting Point)

* **Collections:** `faq_collection`, `chunk_collection`
* **k:** FAQ=10 (early-exit), Chunks=20
* **Gate:**

  * `Z_MIN=0.0`, `MARGIN_DELTA=0.35`
  * `COVERAGE_TOPN=3`, `COVERAGE_MIN=0.40`
  * Verb**atim** only for `kind ∈ {FAQ, RunbookStep}` and **Jaccard ≥ 0.60**
  * **Exact-match** enforcement for error/quoted literals
  * Per-section cap = 3; freshness half-life ≈ 365 days

---

## 11. Reproducibility Bundle (Repo Skeleton)

```
/data/
  faq.jsonl                 # Q→A items (your schema)
  chunks.jsonl              # doc spans (your schema)
  train.jsonl               # retriever triples (query, positive, hard_negatives)
/src/
  rag_gate.py               # multi-signal gate (your calibrated version)
  inference.py              # parallel search + router + gate (verbatim return)
/docs/
  EVAL.md                   # metrics, hold-out protocol, calibration steps
  RUNBOOK.md                # ingestion + chunking rules
```

---

## 12. Limitations & Future Work (Still No LLM)

* Ambiguous queries may **deflect** more often; add better paraphrases in FAQ and strengthen chunk coverage.
* Very long, noisy queries: trim boilerplate; preserve literals.
* Optional: a **tiny lexical filter** (BM25) in front of dense retrieval for huge corpora, or add a **re-ranker** later if latency permits (not required for MVP).

---

## 13. TL;DR Plan (Two Sprints)

* **Sprint 1:** Build two indices; implement router + gate; seed FAQ (10–50), Chunks (50–200); evaluate; tune k/thresholds.
* **Sprint 2:** Fine-tune retriever with hard negatives (2k–10k triples), re-embed, recalibrate; add freshness prior; finalize dashboards.

---

If you want, I can package the **starter datasets** (FAQ, Chunks, Triples) from your examples in your exact schema so you can run end-to-end evaluation immediately.
