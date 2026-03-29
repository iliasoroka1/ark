"""Read side of the hybrid index.

Full-precision cosine (via embedding_cache) + BM25 (via tantivy) merged with
score-weighted RRF, plus graph neighbor expansion.
"""

from __future__ import annotations

import logging
import os
from collections import defaultdict
from datetime import UTC, datetime

import numpy as np
import tantivy

from ark.engine.result import Error, Ok, Result
from ark.engine.embed import Embedding
from ark.engine.embedding_cache import EmbeddingCache
from ark.engine.index import (
    F_ATTRIBUTES, F_CHUNK_ATTRIBUTES, F_CHUNK_BODY, F_CHUNK_ID,
    F_CHUNK_TOKENS, F_CORPUS, F_ID, F_SOURCE_ID,
)
from ark.engine.tokenizer import tokenize_text
from ark.engine.types import SearchErr, SearchHit, SearchParams, SearchScores

log = logging.getLogger(__name__)

_RRF_K = 15.0
_EMBED_WEIGHT = 2.0
_BM25_WEIGHT = 1.5
_GRAPH_HOPS = 2
_GRAPH_MIN_SIM = 0.55
_DECAY_FLOOR = 0.3
_DECAY_HALFLIFE = 365.0
_ACCESS_BOOST_CAP = 1.3
_ACCESS_BOOST_STEP = 0.02
_PRF_DOCS = 3          # number of top cosine hits to extract terms from
_PRF_TERMS = 12        # max terms to extract for pseudo-relevance feedback
_PRF_MIN_SIM = 0.35    # minimum cosine similarity to consider a doc for PRF


class Searcher:
    __slots__ = ("_schema", "_index", "_embedding", "_embed_cache", "_graph_store")

    def __init__(
        self,
        schema: tantivy.Schema,
        index: tantivy.Index,
        embedding: Embedding,
        embed_cache: EmbeddingCache | None = None,
        graph_store=None,
    ) -> None:
        self._schema = schema
        self._index = index
        self._embedding = embedding
        self._embed_cache = embed_cache
        self._graph_store = graph_store

    async def search(
        self,
        query: str,
        corpus: str | None = None,
        source_ids: list[str] | None = None,
        params: SearchParams | None = None,
        expanded_query: str | None = None,
        agent_id: str = "default",
    ) -> Result[list[SearchHit], SearchErr]:
        if params is None:
            params = SearchParams()

        query_vec = None

        # ── Signal 1: Full-precision cosine via embedding cache ──
        cosine_results: list[tuple[str, float]] = []
        match await self._embedding.embed(query):
            case Ok(qv):
                query_vec = qv
                if self._embed_cache is not None:
                    cosine_results = self._embed_cache.search_by_vector(
                        qv, corpus or "", limit=params.num_to_score,
                    )
            case Error(err):
                pass

        # ── Pseudo-relevance feedback (PRF) ──
        # Extract key terms from top cosine hits to auto-expand the query.
        # This bridges vocabulary gaps without hardcoded synonym maps.
        prf_terms = _pseudo_relevance_feedback(
            cosine_results, self._index, self._schema, corpus, source_ids,
            top_k=_PRF_DOCS, max_terms=_PRF_TERMS,
        )
        if prf_terms and not expanded_query:
            expanded_query = prf_terms

        # ── Signal 1b: Enriched cosine ──
        # Re-embed "query + expansion terms" to pull the vector toward
        # the specific vocabulary found in the corpus.
        if expanded_query and expanded_query != query and self._embed_cache is not None:
            enriched = f"{query} {expanded_query}"
            match await self._embedding.embed(enriched):
                case Ok(enr_vec):
                    enr_cosine = self._embed_cache.search_by_vector(
                        enr_vec, corpus or "", limit=params.num_to_score,
                    )
                    existing = dict(cosine_results)
                    for doc_id, sim in enr_cosine:
                        if doc_id not in existing or sim > existing[doc_id]:
                            existing[doc_id] = sim
                    cosine_results = sorted(existing.items(), key=lambda x: x[1], reverse=True)
                case Error(_):
                    pass

        # ── Signal 2: BM25 via tantivy (original query, full weight) ──
        self._index.reload()
        searcher = self._index.searcher()

        bm25_docs: list[tuple[float, object]] = []
        if query.strip():
            bm25_query, _errors = self._index.parse_query_lenient(query, [F_CHUNK_BODY])
            if corpus or source_ids:
                bm25_query = self._wrap_filters(bm25_query, corpus, source_ids)
            bm25_results = searcher.search(bm25_query, limit=params.num_to_score)
            bm25_docs = [(score, searcher.doc(addr)) for score, addr in bm25_results.hits]

        # ── Signal 2b: BM25 on expanded query (half weight) ──
        bm25_expanded_docs: list[tuple[float, object]] = []
        if expanded_query and expanded_query.strip() and expanded_query != query:
            exp_query, _errors = self._index.parse_query_lenient(expanded_query, [F_CHUNK_BODY])
            if corpus or source_ids:
                exp_query = self._wrap_filters(exp_query, corpus, source_ids)
            exp_results = searcher.search(exp_query, limit=params.num_to_score)
            bm25_expanded_docs = [(score, searcher.doc(addr)) for score, addr in exp_results.hits]

        # ── Merge: RRF with score-weighted BM25 ──
        hits = _rrf_merge(cosine_results, bm25_docs, searcher, self._schema,
                          params, self._embed_cache,
                          bm25_expanded_docs=bm25_expanded_docs,
                          agent_id=agent_id)

        # ── Graph expansion ──
        if self._graph_store is not None and self._embed_cache is not None and query_vec is not None:
            hits = _graph_expand(hits, query_vec, self._graph_store, self._embed_cache, params)

        return Ok(hits)

    def _build_bm25_query(self, tokens: list[str]) -> tantivy.Query:
        subqueries = [
            (tantivy.Occur.Should, tantivy.Query.term_query(self._schema, F_CHUNK_TOKENS, tok))
            for tok in tokens
        ]
        return tantivy.Query.boolean_query(subqueries)

    def _wrap_filters(self, query, corpus, source_ids):
        clauses = [(tantivy.Occur.Must, query)]
        if corpus:
            corpus_q = tantivy.Query.term_query(self._schema, F_CORPUS, corpus)
            clauses.append((tantivy.Occur.Must, tantivy.Query.const_score_query(corpus_q, score=0.0)))
        if source_ids:
            sid_subqs = [(tantivy.Occur.Should, tantivy.Query.term_query(self._schema, F_SOURCE_ID, sid)) for sid in source_ids]
            sid_q = tantivy.Query.boolean_query(sid_subqs)
            clauses.append((tantivy.Occur.Must, tantivy.Query.const_score_query(sid_q, score=0.0)))
        return tantivy.Query.boolean_query(clauses)


def _pseudo_relevance_feedback(
    cosine_results: list[tuple[str, float]],
    index: tantivy.Index,
    schema: tantivy.Schema,
    corpus: str | None,
    source_ids: list[str] | None,
    top_k: int = 3,
    max_terms: int = 12,
) -> str | None:
    """Extract key terms from top cosine-similar documents (pseudo-relevance feedback).

    Looks up the body text of the top-K cosine hits in tantivy, tokenizes them,
    and returns the most frequent non-stopword terms. These terms bridge the
    vocabulary gap between the query and relevant documents automatically.
    """
    candidates = [(doc_id, sim) for doc_id, sim in cosine_results[:top_k] if sim >= _PRF_MIN_SIM]
    if not candidates:
        return None

    index.reload()
    searcher = index.searcher()

    bodies: list[str] = []
    for doc_id, _sim in candidates:
        try:
            q = tantivy.Query.term_query(schema, F_ID, doc_id)
            results = searcher.search(q, limit=3)
            for _score, addr in results.hits:
                doc = searcher.doc(addr)
                ca = doc.get_all(F_CHUNK_ATTRIBUTES)
                if ca:
                    v = ca[0]
                    body = v.get("body", "") if isinstance(v, dict) else ""
                    if body:
                        bodies.append(body)
                        break
        except Exception:
            continue

    if not bodies:
        return None

    from collections import Counter
    term_counts: Counter[str] = Counter()
    _STOP = {
        "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "could",
        "should", "may", "might", "shall", "can", "need", "dare", "ought",
        "to", "of", "in", "for", "on", "with", "at", "by", "from", "as",
        "into", "through", "during", "before", "after", "above", "below",
        "between", "out", "off", "over", "under", "again", "further", "then",
        "once", "and", "but", "or", "nor", "not", "so", "yet", "both",
        "each", "few", "more", "most", "other", "some", "such", "no", "only",
        "own", "same", "than", "too", "very", "just", "because", "if", "when",
        "while", "where", "how", "what", "which", "who", "whom", "this",
        "that", "these", "those", "it", "its", "i", "we", "our", "you",
        "your", "he", "she", "they", "them", "their", "all", "any", "up",
    }

    for body in bodies:
        words = tokenize_text(body)
        for w in words:
            if len(w) >= 3 and w not in _STOP:
                term_counts[w] += 1

    if not term_counts:
        return None

    top_terms = [term for term, _count in term_counts.most_common(max_terms)]
    return " ".join(top_terms)


def _doc_field(doc, field):
    vals = doc.get_all(field)
    return str(vals[0]) if vals else None


def _doc_json(doc, field):
    vals = doc.get_all(field)
    if not vals:
        return None
    v = vals[0]
    return v if isinstance(v, dict) else None


def _compute_decay(access_count, last_accessed):
    now = datetime.now(UTC)
    if last_accessed:
        try:
            last_dt = datetime.fromisoformat(last_accessed)
            age_days = (now - last_dt).total_seconds() / 86400.0
        except (ValueError, TypeError):
            age_days = 0.0
    else:
        age_days = _DECAY_HALFLIFE
    decay = max(_DECAY_FLOOR, 1.0 - (age_days / _DECAY_HALFLIFE) * 0.5)
    access_boost = min(_ACCESS_BOOST_CAP, 1.0 + access_count * _ACCESS_BOOST_STEP)
    return decay * access_boost


_EXPANDED_BM25_WEIGHT = 0.5  # weight multiplier for LLM-expanded BM25 terms


def _rrf_merge(cosine_results, bm25_docs, searcher, schema, params, embed_cache=None,
               bm25_expanded_docs=None, agent_id="default"):
    """Merge full-precision cosine results with BM25 results via score-weighted RRF.

    cosine_results: list of (doc_id, cosine_similarity) from embedding cache
    bm25_docs: list of (bm25_score, tantivy_doc) from tantivy search
    bm25_expanded_docs: optional list of (bm25_score, tantivy_doc) from expanded query (half weight)
    """
    scored: dict[str, tuple[SearchScores, float]] = {}  # doc_id → (scores, cosine_sim)

    for rank, (doc_id, cosine_sim) in enumerate(cosine_results):
        rrf = _EMBED_WEIGHT / (_RRF_K + rank + 1)
        scored[doc_id] = (SearchScores(rrf=rrf, embedding=cosine_sim, bm25=0.0), cosine_sim)

    # Score-weighted BM25 fusion
    max_bm25 = bm25_docs[0][0] if bm25_docs else 1.0
    if max_bm25 < 1e-6:
        max_bm25 = 1.0

    bm25_doc_map: dict[str, tuple[float, float]] = {}  # doc_id → (rrf_contribution, raw_bm25)
    for rank, (raw_score, doc) in enumerate(bm25_docs):
        doc_id = _doc_field(doc, F_ID) or ""
        if not doc_id or doc_id in bm25_doc_map:
            continue
        rank_rrf = _BM25_WEIGHT / (_RRF_K + rank + 1)
        score_factor = raw_score / max_bm25
        rrf = rank_rrf * score_factor
        bm25_doc_map[doc_id] = (rrf, raw_score)

    # Merge BM25 into scored
    for doc_id, (bm25_rrf, raw_bm25) in bm25_doc_map.items():
        if doc_id in scored:
            existing, cosine_sim = scored[doc_id]
            merged = SearchScores(
                rrf=existing.rrf + bm25_rrf,
                embedding=existing.embedding,
                bm25=raw_bm25,
            )
            scored[doc_id] = (merged, cosine_sim)
        else:
            scored[doc_id] = (SearchScores(rrf=bm25_rrf, embedding=0.0, bm25=raw_bm25), 0.0)

    # Merge expanded BM25 at reduced weight (vocabulary bridging without noise dominance)
    if bm25_expanded_docs:
        max_exp = bm25_expanded_docs[0][0] if bm25_expanded_docs else 1.0
        if max_exp < 1e-6:
            max_exp = 1.0
        exp_weight = _BM25_WEIGHT * _EXPANDED_BM25_WEIGHT
        seen_exp: set[str] = set()
        for rank, (raw_score, doc) in enumerate(bm25_expanded_docs):
            doc_id = _doc_field(doc, F_ID) or ""
            if not doc_id or doc_id in seen_exp:
                continue
            seen_exp.add(doc_id)
            rank_rrf = exp_weight / (_RRF_K + rank + 1)
            score_factor = raw_score / max_exp
            rrf = rank_rrf * score_factor
            if doc_id in scored:
                existing, cosine_sim = scored[doc_id]
                scored[doc_id] = (
                    SearchScores(rrf=existing.rrf + rrf, embedding=existing.embedding, bm25=existing.bm25),
                    cosine_sim,
                )
            else:
                scored[doc_id] = (SearchScores(rrf=rrf, embedding=0.0, bm25=raw_score), 0.0)

    # Apply decay (skip if ARK_NO_DECAY is set — for deterministic benchmarking)
    if embed_cache is not None and not os.environ.get("ARK_NO_DECAY"):
        doc_ids = [d for d in scored if d]
        if doc_ids:
            meta = embed_cache.get_decay_metadata(doc_ids, agent_id=agent_id)
            for did, (ac, la) in meta.items():
                if did in scored:
                    factor = _compute_decay(ac, la)
                    old_scores, cosine_sim = scored[did]
                    scored[did] = (
                        SearchScores(rrf=old_scores.rrf * factor, embedding=old_scores.embedding, bm25=old_scores.bm25),
                        cosine_sim,
                    )

    # Rank and build hits
    ranked = sorted(scored.items(), key=lambda x: x[1][0].rrf, reverse=True)
    hits = []

    for doc_id, (scores, _cosine) in ranked:
        if scores.rrf < params.min_rrf_score:
            continue
        if scores.bm25 < params.min_bm25_score or scores.embedding < params.min_embedding_score:
            continue

        # Look up doc body from tantivy for display
        body, attrs, chunk_attrs = _lookup_doc_content(doc_id, searcher, schema)

        hits.append(SearchHit(
            doc_id=doc_id,
            chunk_id=f"{doc_id}-0",
            body=body,
            scores=scores,
            attributes=attrs,
            chunk_attributes=chunk_attrs,
        ))
        if len(hits) >= params.num_to_return:
            break

    return hits


def _lookup_doc_content(doc_id, searcher, schema):
    """Look up document body and attributes from tantivy index."""
    try:
        query = tantivy.Query.term_query(schema, F_ID, doc_id)
        results = searcher.search(query, limit=3)
        for _score, addr in results.hits:
            doc = searcher.doc(addr)
            ca = _doc_json(doc, F_CHUNK_ATTRIBUTES)
            if ca and ca.get("body"):
                attrs = _doc_json(doc, F_ATTRIBUTES)
                return ca.get("body", ""), attrs, ca
    except Exception:
        pass
    return "", None, None


def _graph_expand(hits, query_vec, graph_store, embed_cache, params):
    """Expand results by traversing graph edges from top hits."""
    if not hits:
        return hits

    qv = np.array(query_vec, dtype=np.float32)
    qv_norm = np.linalg.norm(qv)
    if qv_norm < 1e-12:
        return hits
    qv = qv / qv_norm

    existing_ids = {h.doc_id for h in hits}
    seed_ids = [h.doc_id for h in hits[:5]]

    candidate_ids: set[str] = set()
    frontier = set(seed_ids)
    visited = set(seed_ids)
    for _hop in range(_GRAPH_HOPS):
        next_frontier: set[str] = set()
        for doc_id in frontier:
            for neighbor_id, _etype, _weight in graph_store.get_neighbors(doc_id, limit=8):
                if neighbor_id not in visited:
                    visited.add(neighbor_id)
                    next_frontier.add(neighbor_id)
                    if neighbor_id not in existing_ids:
                        candidate_ids.add(neighbor_id)
            for pred_id, _etype, _weight in graph_store.get_predecessors(doc_id, limit=8):
                if pred_id not in visited:
                    visited.add(pred_id)
                    next_frontier.add(pred_id)
                    if pred_id not in existing_ids:
                        candidate_ids.add(pred_id)
        frontier = next_frontier

    if not candidate_ids:
        return hits

    vecs = embed_cache.get_many(list(candidate_ids))
    min_rrf = hits[-1].scores.rrf if hits else 0.01
    max_rrf = hits[0].scores.rrf if hits else 0.1

    for doc_id, vec in vecs.items():
        nv = np.array(vec, dtype=np.float32)
        nv_norm = np.linalg.norm(nv)
        if nv_norm < 1e-12:
            continue
        nv = nv / nv_norm
        sim = float(np.dot(qv, nv))
        if sim >= _GRAPH_MIN_SIM:
            rrf_score = min_rrf + (max_rrf - min_rrf) * max(0, (sim - _GRAPH_MIN_SIM) / (1.0 - _GRAPH_MIN_SIM))
            scores = SearchScores(rrf=rrf_score, embedding=sim, bm25=0.0)
            hits.append(SearchHit(
                doc_id=doc_id, chunk_id=f"{doc_id}-graph", body="",
                scores=scores, attributes=None, chunk_attributes=None,
            ))

    hits.sort(key=lambda h: h.scores.rrf, reverse=True)
    return hits[:params.num_to_return]
