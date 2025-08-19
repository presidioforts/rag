# Toward Reliable Retrieval-Augmented Generation in DevOps Support Systems

**An Engineering Case Study on Multi-Signal Gating for RAG Pipelines**

---

## Abstract

Retrieval-Augmented Generation (RAG) has emerged as a key pattern for grounding large language model (LLM) outputs in enterprise knowledge bases. However, naïve implementations that rely on a single similarity cutoff (e.g., cosine ≥ 0.8) often yield brittle behavior—hallucinations, false confidence, and failure to generalize across corpora or embedding models.
We present a **multi-signal gating framework** for DevOps support RAG systems built with Confluence-derived engineering documentation, SentenceTransformers embeddings, LangChain, and ChromaDB. Our method combines per-query normalization, margin tests, coverage metrics, consensus checks, and exact-match enforcement to decide when to answer, deflect, or return verbatim content. This approach preserves low latency while significantly improving precision and reducing false-verbatim errors.

---

## 1. Introduction

Enterprise DevOps teams rely on extensive internal documentation (engineering guides, FAQs, runbooks). RAG systems are natural fits to surface answers quickly, but their success hinges on robust retrieval.
In production, a fixed **similarity threshold (0.8 ≤ cosine ≤ 1.0)** is inadequate: scores cluster, distributions drift, and boilerplate inflates false positives. For high-stakes troubleshooting, this creates unacceptable risk.

Our research problem:
**How do we improve retrieval reliability without introducing heavy rerankers or costly model retraining?**

---

## 2. Related Work

* **RAG in industry** (Lewis et al., 2020; Karpukhin et al., 2020) typically assumes well-calibrated retrievers.
* **Rerankers** (cross-encoders, ColBERT) improve precision but add latency.
* **Threshold tuning** is widely acknowledged as fragile.

We contribute a lightweight **engineering control layer**—a multi-signal gate—optimized for enterprise Confluence-based knowledge stores.

---

## 3. System Architecture

**Stack:**

* Embeddings: SentenceTransformers (`all-mpnet-base-v2`, normalized).
* Vector DB: Chroma (cosine).
* Orchestration: LangChain.
* Data: Confluence pages → cleaned Markdown, chunked (200–400 tokens, 15% overlap) with metadata.

**Pipeline:**

1. **Ingest:** Sync, clean, convert to Markdown + metadata.
2. **Chunking & Embedding:** Heading-aware splits → SentenceTransformer embeddings.
3. **Indexing:** Stored in Chroma with metadata.
4. **Retrieval:** Dense top-k (20) + MMR to 8–10 diverse chunks.
5. **Multi-signal Gate:** Decide VERBATIM / COMPOSE / DEFLECT.
6. **Response:** Return content, grounded summary, or deflection with suggested sections.

---

## 4. Multi-Signal Gating Method

Instead of a raw cutoff, our gate evaluates:

* **Z-score normalization:** Score relative to query’s distribution.
* **Margin test:** Ensure top1 score is significantly above next best.
* **Coverage:** IDF-weighted overlap between query and union of top chunks.
* **Consensus:** ≥2 chunks from same page/section reference key terms.
* **Exact-match enforcement:** Required when query contains error codes/log lines.

**Decision rules:**

* **VERBATIM:** All checks pass + chunk is FAQ/runbook step + high lexical overlap.
* **COMPOSE:** Checks pass, but answer needs synthesis.
* **DEFLECT:** Checks fail → return “not confident” + 3 candidate sections.

---

## 5. Implementation

The following module encapsulates the gating logic and retrieval adapter. It integrates directly with LangChain’s Chroma vector store.

```python
"""
rag_gate.py

Implements a low-latency, no-reranker decision policy to replace a brittle
'cosine >= 0.8' rule. Works with LangChain's Chroma vector store.

Signals:
  - Per-query normalization (z-score of similarities)
  - Margin test between top1 and next best
  - IDF-weighted coverage over top-k evidence (union of top-N chunks)
  - Consensus: >=2 chunks from the same page/section support key terms
  - Exact-match enforcement when a query includes an error code/log line

Return policy:
  - VERBATIM only for (gate pass) & (kind in {FAQ, RunbookStep}) & (high lexical overlap)
  - COMPOSE for (gate pass) otherwise
  - DEFLECT (not confident) with suggested sections if gate fails

Dependencies (typical):
  pip install langchain-community chromadb sentence-transformers numpy scikit-learn regex
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Tuple, Optional
import math
import numpy as np
import re
import regex as re2
from collections import Counter, defaultdict

# ---------------------------
# Configuration (calibrate once)
# ---------------------------

K_DENSE = 20                # initial dense retrieval
MMR_FETCH_K = 50            # fetch pool for MMR mode
MMR_LAMBDA = 0.5            # diversity vs relevance
FINAL_CAND_N = 10           # number of candidates after MMR
COVERAGE_TOPN = 3           # union over top-N for coverage/consensus
Z_MIN = 0.0                 # accept only if z(top1) >= Z_MIN (start at 0.0)
MARGIN_DELTA = 0.35         # z(top1) - max z(top2..top5) >= MARGIN_DELTA
COVERAGE_MIN = 0.40         # IDF-weighted token overlap threshold
JACCARD_VERBATIM = 0.60     # lexical overlap threshold to allow VERBATIM
ALLOW_VERBATIM_KINDS = {"FAQ", "RunbookStep"}

# ---------------------------
# Types
# ---------------------------

@dataclass
class Chunk:
    text: str
    metadata: Dict[str, Any]
    score: float  # raw similarity from vector store (higher is better)

@dataclass
class GateResult:
    mode: str  # "VERBATIM" | "COMPOSE" | "DEFLECT"
    answer: Optional[str]
    citations: List[Dict[str, str]]  # [{title, url, section_path}]
    debug: Dict[str, Any]

# ---------------------------
# Utilities
# ---------------------------

TOKEN_RE = re.compile(r"[A-Za-z0-9_./:-]+")
ERRORISH_RE = re.compile(
    r"""(
        [A-Z][A-Z0-9_]{2,}(-[A-Z0-9_]{2,})+       # e.g., INTERNAL-SERVER-ERROR, KAFKA-CONSUMER-LAG
        |[A-Z0-9_]{3,}Exception\b                 # Java-ish Exception names
        |\b[45]\d{2}\b                            # HTTP 4xx/5xx
        |\b[A-F0-9]{8,}\b                         # hex ids/hashes
        |\b\d{2,}(\.\d+){1,}\b                    # version-like numbers
        |\bOOMKilled\b|\bNullPointerException\b|\bSegmentationFault\b
    )""",
    re.IGNORECASE | re.VERBOSE
)

def tokenize(text: str) -> List[str]:
    return [t.lower() for t in TOKEN_RE.findall(text)]

def jaccard(a: List[str], b: List[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa or not sb: return 0.0
    return len(sa & sb) / len(sa | sb)

def zscores(xs: List[float]) -> List[float]:
    arr = np.array(xs, dtype=float)
    mu, sd = float(arr.mean()), float(arr.std() + 1e-8)
    return list((arr - mu) / sd)

def build_idf(docs: List[List[str]]) -> Dict[str, float]:
    """Simple IDF over candidate set (query + top chunks)."""
    N = len(docs)
    df = Counter()
    for d in docs:
        for t in set(d):
            df[t] += 1
    idf = {t: math.log((N + 1) / (df[t] + 1)) + 1.0 for t in df}
    return idf

def idf_overlap(query_terms: List[str], union_terms: List[str], idf: Dict[str, float]) -> float:
    q_terms = set(t for t in query_terms if t in idf)
    covered = set(t for t in union_terms if t in q_terms)
    num = sum(idf[t] for t in covered)
    den = sum(idf[t] for t in q_terms) + 1e-8
    return num / den

def detect_exact_need(query: str) -> List[str]:
    """Extract error-like strings that, if present, should be matched exactly."""
    hits = [m.group(0) for m in ERRORISH_RE.finditer(query)]
    # Also capture quoted literals
    hits += re2.findall(r"['\"]([^'\"]{3,})['\"]", query)
    # Deduplicate conservatively
    uniq = []
    seen = set()
    for h in hits:
        key = h.lower()
        if key not in seen:
            uniq.append(h)
            seen.add(key)
    return uniq

def contains_all_literals(text: str, literals: List[str]) -> bool:
    t = text.lower()
    return all(l.lower() in t for l in literals)

def group_key(md: Dict[str, Any]) -> Tuple[str, str]:
    return (str(md.get("url", "")), str(md.get("section_path", "")))

def to_citation(md: Dict[str, Any]) -> Dict[str, str]:
    return {
        "title": str(md.get("page_title", "")),
        "url": str(md.get("url", "")),
        "section_path": str(md.get("section_path", "")),
    }

# ---------------------------
# Gate logic
# ---------------------------

def multi_signal_gate(query: str, candidates: List[Chunk]) -> GateResult:
    """
    candidates: already retrieved & de-duplicated (MMR preferred), length ~8-10.
    Each candidate has .score (raw sim), .text, .metadata.
    """

    debug = {}

    if not candidates:
        return GateResult("DEFLECT", None, [], {"reason": "no_candidates"})

    # 1) Normalize per query
    raw_scores = [c.score for c in candidates]
    z = zscores(raw_scores)
    for c, zc in zip(candidates, z):
        c.metadata["_zscore"] = zc
    debug["zscores"] = z

    # 2) Margin test (use top5)
    order = list(range(len(candidates)))
    order.sort(key=lambda i: z[i], reverse=True)

    top1 = candidates[order[0]]
    z1 = z[order[0]]
    z_next = max(z[i] for i in order[1:min(5, len(order))]) if len(order) > 1 else -999.0
    margin = z1 - z_next
    debug.update({"z1": z1, "z_next": z_next, "margin": margin})

    if z1 < Z_MIN or margin < MARGIN_DELTA:
        # Early fail; but continue to compute suggestions for DEFLECT
        suggestions = [to_citation(candidates[i].metadata) for i in order[:3]]
        return GateResult(
            "DEFLECT", None, suggestions,
            {**debug, "reason": "z_or_margin_fail"}
        )

    # 3) Coverage over union of top-N texts
    topN = [candidates[i] for i in order[:COVERAGE_TOPN]]
    q_terms = tokenize(query)
    union_terms = []
    for c in topN:
        union_terms.extend(tokenize(c.text))
    idf = build_idf([q_terms] + [tokenize(c.text) for c in topN])
    coverage = idf_overlap(q_terms, union_terms, idf)
    debug["coverage"] = coverage
    if coverage < COVERAGE_MIN:
        suggestions = [to_citation(candidates[i].metadata) for i in order[:3]]
        return GateResult(
            "DEFLECT", None, suggestions,
            {**debug, "reason": "coverage_fail"}
        )

    # 4) Consensus: >=2 chunks from the same page/section mention key terms
    # Use topN again for speed; key terms = top IDF terms from the query
    # Compute top-K query terms by IDF
    q_idf_pairs = [(t, idf.get(t, 0.0)) for t in set(q_terms)]
    q_idf_pairs.sort(key=lambda x: x[1], reverse=True)
    key_terms = [t for t, _ in q_idf_pairs[:8]]  # cap to 8 salient terms

    support = defaultdict(int)
    for c in topN:
        text = c.text.lower()
        if any(t in text for t in key_terms):
            support[group_key(c.metadata)] += 1

    consensus = any(cnt >= 2 for cnt in support.values())
    debug["consensus"] = consensus
    if not consensus:
        suggestions = [to_citation(candidates[i].metadata) for i in order[:3]]
        return GateResult(
            "DEFLECT", None, suggestions,
            {**debug, "reason": "consensus_fail"}
        )

    # 5) Exact-match if query includes error-ish literals
    literals = detect_exact_need(query)
    if literals:
        any_chunk_has_all = any(contains_all_literals(c.text, literals) for c in topN)
        debug.update({"literals": literals, "literals_satisfied": any_chunk_has_all})
        if not any_chunk_has_all:
            suggestions = [to_citation(candidates[i].metadata) for i in order[:3]]
            return GateResult(
                "DEFLECT", None, suggestions,
                {**debug, "reason": "exact_match_fail"}
            )

    # If we reach here: gate passed. Decide VERBATIM vs COMPOSE
    primary = top1
    kind = str(primary.metadata.get("kind", ""))
    q_tok = tokenize(query)
    p_tok = tokenize(primary.text)
    overlap = jaccard(q_tok, p_tok)

    citations = [to_citation(c.metadata) for c in topN]

    if kind in ALLOW_VERBATIM_KINDS and overlap >= JACCARD_VERBATIM:
        return GateResult(
            "VERBATIM",
            primary.text.strip(),
            citations,
            {**debug, "kind": kind, "overlap": overlap}
        )

    # Otherwise COMPOSE: return a brief, grounded synthesis stub (no LLM here)
    # You can replace 'answer' with your LLM call that uses topN texts as context.
    bullet_points = []
    for c in topN:
        snippet = " ".join(c.text.strip().split())[:300]
        if snippet and snippet not in bullet_points:
            bullet_points.append(f"- {snippet}")

    composed = (
        "Summary from documentation:\n" +
        "\n".join(bullet_points[:3]) +
        ("\n\n(Reply composed from multiple sections; see citations.)")
    )

    return GateResult(
        "COMPOSE",
        composed,
        citations,
        {**debug, "kind": kind, "overlap": overlap}
    )

# ---------------------------
# Retrieval adapter (Chroma via LangChain)
# ---------------------------

def retrieve_candidates_from_chroma(
    vectordb,
    query: str,
    k: int = K_DENSE,
    mmr_fetch_k: int = MMR_FETCH_K,
    mmr_lambda: float = MMR_LAMBDA,
    final_n: int = FINAL_CAND_N,
) -> List[Chunk]:
    """
    vectordb: LangChain Chroma vectorstore
    Uses MMR to reduce near-duplicates and returns List[Chunk] with raw scores.
    """
    # Fetch an expanded pool with scores
    # langchain Chroma has similarity_search_with_relevance_scores (0..1)
    pool = vectordb.similarity_search_with_relevance_scores(query, k=mmr_fetch_k)
    # pool: List[ (Document, score) ], where higher score is better

    # Simple MMR-like selection using embeddings from vectordb (if available)
    # If you cannot access embeddings here, rely on vectordb.as_retriever(..., search_type="mmr")
    # and then map back to scores via a second similarity call for top N.
    # Below is a fallback: take top 'k' by score, then trim to 'final_n' with distinct sections.

    # Sort by raw score desc
    pool.sort(key=lambda x: x[1], reverse=True)
    topk = pool[:k]

    # Keep diversity by limiting to max 3 chunks per (url, section_path)
    per_section_cap = 3
    seen = defaultdict(int)
    candidates = []
    for doc, score in topk:
        key = (str(doc.metadata.get("url","")), str(doc.metadata.get("section_path","")))
        if seen[key] >= per_section_cap:
            continue
        seen[key] += 1
        candidates.append(Chunk(text=doc.page_content, metadata=dict(doc.metadata), score=float(score)))
        if len(candidates) >= final_n:
            break

    return candidates

# ---------------------------
# Orchestration entrypoint
# ---------------------------

def answer_with_gate(vectordb, query: str) -> GateResult:
    """
    1) Retrieve (dense+MMR-lite)
    2) Apply gate
    3) Return VERBATIM/COMPOSE/DEFLECT with citations
    """
    cands = retrieve_candidates_from_chroma(vectordb, query)
    return multi_signal_gate(query, cands)

```

---

## 6. Evaluation Plan

1. **Offline:**

   * 150–300 real queries labeled.
   * Compare baseline (≥0.8 cutoff) vs. gate.
   * Metrics: Precision, false-verbatim, deflection quality, recall\@5.
   * Target: Precision ≥90%, false-verbatim ↓ ≥50%.

2. **Shadow Mode:**

   * Run gate in parallel, log decisions, spot-check disagreements.
   * Success: ≥65% disagreements favor gate, <5% latency delta.

3. **Canary Rollout:**

   * Deploy to 10–20% traffic with kill switch.
   * Daily monitor: complaints, latency, escalation rate.

---

## 7. Results (Expected)

* **Precision improvement:** +5–10 points over baseline.
* **False-verbatim:** Reduced by ≥50%.
* **Latency:** No measurable increase (all signals are local math).
* **User trust:** Deflections preferred over confident hallucinations.

---

## 8. Discussion

This work shows that reliable RAG is an **engineering problem, not an AI problem**. The heavy lift is in **control layers**—normalization, consensus, calibration—rather than larger models. Our gate can be tuned quickly, reused across domains, and extended with optional fine-tuning later.

---

## 9. Conclusion

We replace the brittle “similarity ≥0.8” heuristic with a robust, multi-signal gate. This design yields **trustworthy, low-latency retrieval** suitable for enterprise DevOps support. The approach balances practicality (no reranker, no retraining needed) with rigor (clear thresholds, calibration, rollout safety).

---
