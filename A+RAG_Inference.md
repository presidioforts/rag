Excellent catches. I’ve applied your four “quick fixes” and tightened a couple of small edges while keeping everything else identical. Summary of changes:

* **Gate logic fix:** now requires `(consensus OR literals_ok) AND (if literals exist ⇒ literals_ok)`.
* **Safer Jaccard:** defend against empty unions without masking other cases.
* **zscores() wrapped:** full try/except around NumPy path; returns zeros on failure.
* **Robust freshness parsing:** safer ISO/Z handling; preserves epoch support; returns neutral weight on parse failure.

Here’s the **drop-in, executable** `rag_gate.py` (A+ core, no reranker).

```python
"""
rag_gate.py

Implements a low-latency, no-reranker decision policy to replace a brittle
'cosine >= 0.8' rule. Designed for LangChain's Chroma vector store.

Signals:
  - Per-query normalization (z-score of similarities)
  - Margin test between top1 and next best
  - IDF-weighted coverage over top-k evidence (union of top-N chunks)
  - Consensus: >=2 chunks from the same page/section with sufficient term overlap
  - Exact-match enforcement when a query includes an error code/log line

Return policy:
  - VERBATIM only for (gate pass) & (kind in {FAQ, RunbookStep}) & (high lexical overlap)
  - COMPOSE for (gate pass) otherwise
  - DEFLECT (not confident) with suggested sections if gate fails

Notes:
  - This module implements "MMR-lite" diversity via per-section capping.
    To switch to true MMR, use LangChain's MMR retriever and feed the gated logic
    with those candidates (see retrieval adapter docstring).

Dependencies (typical):
  pip install langchain-community chromadb sentence-transformers numpy regex
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Tuple, Optional
import math
import numpy as np
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
import logging

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Configuration (calibrate once)
# -----------------------------------------------------------------------------
K_DENSE = 20                 # initial dense retrieval
MMR_FETCH_K = 50             # fetch pool (if using built-in MMR retriever)
MMR_LAMBDA = 0.5             # diversity vs relevance (for true MMR retriever)
FINAL_CAND_N = 10            # number of candidates forwarded to gating
COVERAGE_TOPN = 3            # union over top-N for coverage/consensus
Z_MIN = 0.0                  # require z(top1) >= Z_MIN (start at 0.0)
MARGIN_DELTA = 0.35          # z(top1) - max z(top2..top5) >= MARGIN_DELTA
COVERAGE_MIN = 0.40          # IDF-weighted token overlap threshold
JACCARD_VERBATIM = 0.60      # lexical overlap threshold to allow VERBATIM
ALLOW_VERBATIM_KINDS = {"FAQ", "RunbookStep"}

# Freshness weighting (light prior, optional)
FRESHNESS_HALF_LIFE_DAYS = 365  # down-weight chunks older than this scale
FRESHNESS_WEIGHTING = True      # set False to disable

# Security limits
MAX_QUERY_LENGTH = 10000        # Prevent DoS from extremely long queries
MAX_CANDIDATES = 100            # Prevent memory exhaustion
MIN_CONTENT_TOKENS = 3          # Queries with fewer tokens are likely under-specified

# -----------------------------------------------------------------------------
# Types
# -----------------------------------------------------------------------------
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

# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------
TOKEN_RE = re.compile(r"[A-Za-z0-9_./:-]+")
ERRORISH_RE = re.compile(
    r"""(
        [A-Z][A-Z0-9_]{2,}(-[A-Z0-9_]{2,})+       # INTERNAL-SERVER-ERROR, KAFKA-CONSUMER-LAG
        |[A-Z0-9_]{3,}Exception\b                 # Java-ish Exception names
        |\b[45]\d{2}\b                            # HTTP 4xx/5xx
        |\b[A-F0-9]{8,}\b                         # hex ids/hashes
        |\b\d{2,}(\.\d+){1,}\b                    # version-like numbers
        |\bOOMKilled\b|\bNullPointerException\b|\bSegmentationFault\b
    )""",
    re.IGNORECASE | re.VERBOSE
)

def _safe_str(x: Any) -> str:
    try:
        return str(x)
    except Exception:
        return ""

def validate_input(query: str, candidates: List[Chunk]) -> None:
    if not isinstance(query, str):
        raise ValueError("Query must be a string")
    if len(query) > MAX_QUERY_LENGTH:
        raise ValueError(f"Query too long: {len(query)} > {MAX_QUERY_LENGTH}")
    if not isinstance(candidates, list):
        raise ValueError("Candidates must be a list")
    if len(candidates) > MAX_CANDIDATES:
        raise ValueError(f"Too many candidates: {len(candidates)} > {MAX_CANDIDATES}")
    for i, c in enumerate(candidates):
        if not isinstance(c, Chunk):
            raise ValueError(f"Candidate {i} is not a Chunk")

def tokenize(text: str) -> List[str]:
    if not isinstance(text, str) or not text.strip():
        return []
    try:
        return [t.lower() for t in TOKEN_RE.findall(text)]
    except Exception as e:
        logger.warning(f"Tokenization failed: {e}")
        return []

def jaccard(a: List[str], b: List[str]) -> float:
    if not a or not b:
        return 0.0
    try:
        sa, sb = set(a), set(b)
        if not sa or not sb:
            return 0.0
        union_size = len(sa | sb)
        return (len(sa & sb) / union_size) if union_size > 0 else 0.0
    except Exception:
        return 0.0

def zscores(xs: List[float]) -> List[float]:
    if not xs:
        return []
    if len(xs) == 1:
        return [0.0]
    try:
        arr = np.array(xs, dtype=float)
        if not np.all(np.isfinite(arr)):
            logger.warning("Non-finite similarity score(s) encountered; replacing with 0.0")
            arr = np.where(np.isfinite(arr), arr, 0.0)
        mu = float(arr.mean())
        sd = float(arr.std())
        if sd == 0.0:
            return [0.0] * len(xs)
        return list((arr - mu) / sd)
    except Exception as e:
        logger.warning(f"Z-score calculation failed: {e}")
        return [0.0] * len(xs)

def build_idf(docs: List[List[str]]) -> Dict[str, float]:
    if not docs:
        return {}
    N = len(docs)
    df = Counter()
    for d in docs:
        if not d:
            continue
        for t in set(d):
            df[t] += 1
    return {t: math.log((N + 1) / (df[t] + 1)) + 1.0 for t in df}

def idf_overlap(query_terms: List[str], union_terms: List[str], idf: Dict[str, float]) -> float:
    if not query_terms or not union_terms or not idf:
        return 0.0
    q_terms = set(query_terms) & set(idf.keys())
    if not q_terms:
        return 0.0
    covered = set(union_terms) & q_terms
    num = sum(idf[t] for t in covered)
    den = sum(idf[t] for t in q_terms)
    if den == 0.0:
        return 0.0
    return num / den

def detect_exact_need(query: str) -> List[str]:
    if not query or len(query) > MAX_QUERY_LENGTH:
        return []
    hits = [m.group(0) for m in ERRORISH_RE.finditer(query)]
    # capture quoted literals
    quoted = re.compile(r"['\"]([^'\"]{3,})['\"]")
    hits += [m.group(1) for m in quoted.finditer(query)]
    uniq, seen = [], set()
    for h in hits:
        key = h.lower()
        if key and key not in seen:
            uniq.append(h)
            seen.add(key)
    return uniq[:10]  # cap to avoid abuse

def contains_all_literals(text: str, literals: List[str]) -> bool:
    if not text or not literals:
        return len(literals) == 0
    t = text.lower()
    return all(l.lower() in t for l in literals if l)

def group_key(md: Dict[str, Any]) -> Tuple[str, str]:
    return (_safe_str(md.get("url", "")), _safe_str(md.get("section_path", "")))

def to_citation(md: Dict[str, Any]) -> Dict[str, str]:
    return {
        "title": _safe_str(md.get("page_title", "")),
        "url": _safe_str(md.get("url", "")),
        "section_path": _safe_str(md.get("section_path", "")),
    }

def _parse_updated_at(ts: Any) -> Optional[datetime]:
    try:
        if isinstance(ts, (int, float)):
            return datetime.fromtimestamp(float(ts), tz=timezone.utc)
        if isinstance(ts, str):
            s = ts.strip()
            if s.endswith("Z"):
                s = s[:-1] + "+00:00"
            return datetime.fromisoformat(s)
        return None
    except Exception:
        # Fallback: try naive parse; if still fails, return None
        try:
            s = str(ts).replace("Z", "+00:00")
            return datetime.fromisoformat(s)
        except Exception:
            return None

def _freshness_weight(md: Dict[str, Any]) -> float:
    """Light prior: down-weight old sections by half-life. 1.0 = fresh."""
    if not FRESHNESS_WEIGHTING:
        return 1.0
    updated = _parse_updated_at(md.get("updated_at"))
    if not updated:
        return 1.0
    try:
        age_days = max(0.0, (datetime.now(tz=timezone.utc) - updated).total_seconds() / 86400.0)
        lam = math.log(2) / max(1.0, float(FRESHNESS_HALF_LIFE_DAYS))
        return float(math.exp(-lam * age_days))
    except Exception:
        return 1.0

# -----------------------------------------------------------------------------
# Gate logic
# -----------------------------------------------------------------------------
def multi_signal_gate(query: str, candidates: List[Chunk]) -> GateResult:
    """
    candidates: already retrieved & de-duplicated (MMR-lite), length ~8–10.
    Each candidate has .score (raw sim), .text, .metadata.

    Gate = (z ok ∧ margin ok ∧ coverage ok ∧ (consensus OR (literals present & satisfied)))
    Then:
      - VERBATIM if kind in {FAQ, RunbookStep} & high lexical overlap
      - else COMPOSE
      - else DEFLECT
    """
    debug: Dict[str, Any] = {}
    try:
        validate_input(query, candidates)

        if not candidates:
            return GateResult("DEFLECT", None, [], {"reason": "no_candidates"})

        # Freshness-adjust raw scores (light prior)
        raw_scores = []
        for c in candidates:
            w = _freshness_weight(c.metadata)
            raw_scores.append(float(c.score) * w)
        z = zscores(raw_scores)
        if not z or len(z) != len(candidates):
            return GateResult("DEFLECT", None, [], {"reason": "zscore_error"})

        # Rank by z-score
        order = list(range(len(candidates)))
        order.sort(key=lambda i: z[i], reverse=True)

        top1_idx = order[0]
        z1 = z[top1_idx]
        next_indices = order[1:5]  # up to next 4 items
        if next_indices:
            z_next = max(z[i] for i in next_indices)
            margin = z1 - z_next
        else:
            z_next = -999.0
            margin = 999.0

        debug.update({"zscores": z, "z1": z1, "z_next": z_next, "margin": margin})

        if z1 < Z_MIN or margin < MARGIN_DELTA:
            suggestions = [to_citation(candidates[i].metadata) for i in order[:min(3, len(order))]]
            out = GateResult("DEFLECT", None, suggestions, {**debug, "reason": "z_or_margin_fail"})
            logger.info("rag_gate_decision", extra={"mode": out.mode, **out.debug})
            return out

        # Coverage over union of top-N texts
        topN_count = min(COVERAGE_TOPN, len(order))
        topN = [candidates[i] for i in order[:topN_count]]

        q_terms = tokenize(query)
        if sum(t.isalnum() for t in q_terms) < MIN_CONTENT_TOKENS:
            out = GateResult("DEFLECT", None, [], {**debug, "reason": "query_too_short"})
            logger.info("rag_gate_decision", extra={"mode": out.mode, **out.debug})
            return out

        union_terms: List[str] = []
        cached_tokens: Dict[int, List[str]] = {}
        for idx, c in zip(order[:topN_count], topN):
            toks = tokenize(c.text)
            cached_tokens[idx] = toks
            union_terms.extend(toks)

        idf = build_idf([q_terms] + [cached_tokens[i] for i in order[:topN_count]])
        coverage = idf_overlap(q_terms, union_terms, idf)
        debug["coverage"] = coverage
        if coverage < COVERAGE_MIN:
            suggestions = [to_citation(candidates[i].metadata) for i in order[:min(3, len(order))]]
            out = GateResult("DEFLECT", None, suggestions, {**debug, "reason": "coverage_fail"})
            logger.info("rag_gate_decision", extra={"mode": out.mode, **out.debug})
            return out

        # Salient terms (top by IDF)
        q_idf_pairs = [(t, idf.get(t, 0.0)) for t in set(q_terms)]
        q_idf_pairs.sort(key=lambda x: x[1], reverse=True)
        key_terms = [t for t, _ in q_idf_pairs[:8]]  # cap to 8 salient terms

        # Consensus: require >= 2 chunks from the same section, each covering >= 1/3 of salient terms (min 1)
        min_terms_per_chunk = max(1, math.ceil(len(key_terms) / 3)) if key_terms else 1
        support = defaultdict(int)
        for i in order[:topN_count]:
            text = (candidates[i].text or "").lower()
            matched = sum(1 for t in key_terms if t in text)
            if matched >= min_terms_per_chunk:
                support[group_key(candidates[i].metadata)] += 1
        consensus = any(cnt >= 2 for cnt in support.values())
        debug["consensus"] = consensus

        # Exact literal requirement (if present in query)
        literals = detect_exact_need(query)
        literals_ok = any(contains_all_literals((candidates[i].text or ""), literals) for i in order[:topN_count]) if literals else False
        debug.update({"literals": literals, "literals_satisfied": literals_ok})

        # ---- FIXED GATE LOGIC ----
        # Gate should pass iff:
        #   (consensus OR literals_ok) AND (if literals exist, literals_ok must be True)
        gate_pass = (consensus or literals_ok) and (not bool(literals) or literals_ok)
        if not gate_pass:
            reason = "exact_match_fail" if (literals and not literals_ok) else "consensus_fail"
            suggestions = [to_citation(candidates[i].metadata) for i in order[:min(3, len(order))]]
            out = GateResult("DEFLECT", None, suggestions, {**debug, "reason": reason})
            logger.info("rag_gate_decision", extra={"mode": out.mode, **out.debug})
            return out

        # Gate passed: decide VERBATIM vs COMPOSE
        primary = candidates[top1_idx]
        kind = _safe_str(primary.metadata.get("kind", ""))
        q_tok = q_terms
        p_tok = tokenize(primary.text)
        overlap = jaccard(q_tok, p_tok)
        citations = [to_citation(candidates[i].metadata) for i in order[:topN_count]]

        if kind in ALLOW_VERBATIM_KINDS and overlap >= JACCARD_VERBATIM:
            out = GateResult(
                "VERBATIM",
                (primary.text or "").strip(),
                citations,
                {**debug, "kind": kind, "overlap": overlap}
            )
            logger.info("rag_gate_decision", extra={"mode": out.mode, **out.debug})
            return out

        # Compose a brief grounded synthesis (placeholder: bullet summary).
        bullet_points: List[str] = []
        for i in order[:topN_count]:
            snippet = " ".join((candidates[i].text or "").strip().split())
            if snippet:
                bullet_points.append(f"- {snippet[:300]}")
        if not bullet_points:
            out = GateResult("DEFLECT", None, citations, {**debug, "reason": "no_valid_content"})
            logger.info("rag_gate_decision", extra={"mode": out.mode, **out.debug})
            return out

        composed = (
            "Summary from documentation:\n" +
            "\n".join(bullet_points[:3]) +
            ("\n\n(Reply composed from multiple sections; see citations.)")
        )
        out = GateResult(
            "COMPOSE",
            composed,
            citations,
            {**debug, "kind": kind, "overlap": overlap}
        )
        logger.info("rag_gate_decision", extra={"mode": out.mode, **out.debug})
        return out

    except Exception as e:
        logger.error(f"Gate processing failed: {e}")
        return GateResult("DEFLECT", None, [], {"reason": "internal_error", "error": str(e)})

# -----------------------------------------------------------------------------
# Retrieval adapter (Chroma via LangChain)
# -----------------------------------------------------------------------------
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

    Diversity:
      - This implementation uses "MMR-lite" via per-section capping (<=3 per (url, section_path)).
      - To enable true MMR diversity, prefer:
          retr = vectordb.as_retriever(search_type="mmr",
                                       search_kwargs={"k": k, "fetch_k": mmr_fetch_k, "lambda_mult": mmr_lambda})
          docs = retr.get_relevant_documents(query)
        Then map docs back to scores with similarity_search_with_relevance_scores for the final_n.
    """
    try:
        if not isinstance(query, str) or not query.strip():
            return []
        if len(query) > MAX_QUERY_LENGTH:
            logger.warning(f"Query truncated from {len(query)} to {MAX_QUERY_LENGTH}")
            query = query[:MAX_QUERY_LENGTH]

        # Fetch an expanded pool with scores
        pool = vectordb.similarity_search_with_relevance_scores(query, k=min(mmr_fetch_k, MAX_CANDIDATES))
        if not pool:
            return []

        # Sort by raw score desc and take top-k
        pool.sort(key=lambda x: x[1], reverse=True)
        topk = pool[:min(k, len(pool))]

        # Keep diversity by limiting to max 3 chunks per (url, section_path)
        per_section_cap = 3
        seen = defaultdict(int)
        candidates: List[Chunk] = []

        for doc, score in topk:
            key = group_key(getattr(doc, "metadata", {}) or {})
            if seen[key] >= per_section_cap:
                continue
            seen[key] += 1

            s = float(score) if isinstance(score, (int, float)) and np.isfinite(score) else 0.0
            candidates.append(Chunk(
                text=_safe_str(getattr(doc, "page_content", "")),
                metadata=dict(getattr(doc, "metadata", {}) or {}),
                score=s
            ))
            if len(candidates) >= min(final_n, MAX_CANDIDATES):
                break

        return candidates

    except Exception as e:
        logger.error(f"Retrieval failed: {e}")
        return []

# -----------------------------------------------------------------------------
# Orchestration entrypoint
# -----------------------------------------------------------------------------
def answer_with_gate(vectordb, query: str) -> GateResult:
    """
    1) Retrieve (dense + MMR-lite)
    2) Apply gate
    3) Return VERBATIM / COMPOSE / DEFLECT with citations
    """
    try:
        if not isinstance(query, str):
            return GateResult("DEFLECT", None, [], {"reason": "invalid_query_type"})
        q = query.strip()
        if not q:
            return GateResult("DEFLECT", None, [], {"reason": "empty_query"})

        cands = retrieve_candidates_from_chroma(vectordb, q)
        return multi_signal_gate(q, cands)

    except Exception as e:
        logger.error(f"Answer generation failed: {e}")
        return GateResult("DEFLECT", None, [], {"reason": "system_error", "error": str(e)})
```

If you want, I can also include a **tiny smoke test** snippet (separate file) to sanity-check each decision path locally against mocked chunks.
