"""Read side of the hybrid index. Embedding + BM25 search merged via RRF."""

from __future__ import annotations

import logging
from collections import defaultdict
from datetime import UTC, datetime

import tantivy

import numpy as np

from ark.engine.result import Error, Ok, Result
from ark.engine.embed import Embedding
from ark.engine.embedding_cache import EmbeddingCache
from ark.engine.index import (
    F_ATTRIBUTES, F_CHUNK_ATTRIBUTES, F_CHUNK_ID,
    F_CHUNK_TOKENS, F_CORPUS, F_ID, F_SOURCE_ID,
)
from ark.engine.tokenizer import binarize_embedding, pluralize_expand, tokenize_text
from ark.engine.types import SearchErr, SearchHit, SearchParams, SearchScores

log = logging.getLogger(__name__)

_RRF_K = 15.0
_EMBED_WEIGHT = 2.0
_BM25_WEIGHT = 1.5
_PRF_TOP_K = 3        # top embedding results to extract PRF tokens from
_PRF_SCORE = 0.1      # const_score for PRF expansion tokens (low to avoid noise)
_GRAPH_HOPS = 2       # max hops for graph neighbor expansion
_GRAPH_MIN_SIM = 0.55 # min cosine similarity for graph neighbors to be injected
_DECAY_FLOOR = 0.3
_DECAY_HALFLIFE = 365.0
_ACCESS_BOOST_CAP = 1.3
_ACCESS_BOOST_STEP = 0.02


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
    ) -> Result[list[SearchHit], SearchErr]:
        if params is None:
            params = SearchParams()

        self._index.reload()
        searcher = self._index.searcher()

        query_vec = None
        match await self._embedding.embed(query):
            case Ok(qv):
                query_vec = qv
                embed_query = self._build_embedding_query(qv)
                if corpus or source_ids:
                    embed_query = self._wrap_filters(embed_query, corpus, source_ids)
                embed_results = searcher.search(embed_query, limit=params.num_to_score)
                embed_docs = [(score, searcher.doc(addr)) for score, addr in embed_results.hits]
            case Error(err):
                embed_docs = []

        # BM25 with plural expansion
        bm25_tokens = pluralize_expand(tokenize_text(query))
        if bm25_tokens:
            bm25_query = self._build_bm25_query(bm25_tokens)
            if corpus or source_ids:
                bm25_query = self._wrap_filters(bm25_query, corpus, source_ids)
            bm25_results = searcher.search(bm25_query, limit=params.num_to_score)
            bm25_docs = [(score, searcher.doc(addr)) for score, addr in bm25_results.hits]
        else:
            bm25_docs = []

        hits = _rrf_merge(embed_docs, bm25_docs, params, self._embed_cache)

        # Graph expansion: traverse edges from top results to find related docs
        if self._graph_store is not None and self._embed_cache is not None and query_vec is not None:
            hits = _graph_expand(hits, query_vec, self._graph_store, self._embed_cache, params)

        return Ok(hits)

    def _build_embedding_query(self, embedding: list[float]) -> tantivy.Query:
        dims = len(embedding)
        score = 1.0 / dims
        tokens = binarize_embedding(embedding)
        subqueries = [
            (tantivy.Occur.Should, tantivy.Query.const_score_query(
                tantivy.Query.term_query(self._schema, F_CHUNK_TOKENS, tok), score=score,
            ))
            for tok in tokens
        ]
        return tantivy.Query.boolean_query(subqueries)

    def _build_bm25_query(self, tokens: list[str]) -> tantivy.Query:
        subqueries = [
            (tantivy.Occur.Should, tantivy.Query.term_query(self._schema, F_CHUNK_TOKENS, tok))
            for tok in tokens
        ]
        return tantivy.Query.boolean_query(subqueries)

    def _build_prf_bm25_query(self, query_tokens: list[str], prf_tokens: set[str]) -> tantivy.Query:
        """BM25 query with original tokens at full weight + PRF tokens at low const_score."""
        subqueries = []
        seen: set[str] = set()
        for tok in query_tokens:
            if tok not in seen:
                seen.add(tok)
                subqueries.append(
                    (tantivy.Occur.Should, tantivy.Query.term_query(self._schema, F_CHUNK_TOKENS, tok))
                )
        for tok in prf_tokens:
            if tok not in seen:
                seen.add(tok)
                subqueries.append(
                    (tantivy.Occur.Should, tantivy.Query.const_score_query(
                        tantivy.Query.term_query(self._schema, F_CHUNK_TOKENS, tok),
                        score=_PRF_SCORE,
                    ))
                )
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


def _rrf_merge(embed_docs, bm25_docs, params, embed_cache=None):
    scored = {}
    for rank, (raw_score, doc) in enumerate(embed_docs):
        cid = _doc_field(doc, F_CHUNK_ID)
        if not cid:
            continue
        rrf = _EMBED_WEIGHT / (_RRF_K + rank + 1)
        scored[cid] = (SearchScores(rrf=rrf, embedding=raw_score, bm25=0.0), doc)

    # Score-weighted BM25 fusion: use both rank AND raw score magnitude.
    # This preserves the information that a doc matching "bug" (score=3.99)
    # is much more relevant than one matching only "and" (score=1.65).
    max_bm25 = bm25_docs[0][0] if bm25_docs else 1.0
    if max_bm25 < 1e-6:
        max_bm25 = 1.0

    for rank, (raw_score, doc) in enumerate(bm25_docs):
        cid = _doc_field(doc, F_CHUNK_ID)
        if not cid:
            continue
        # Combine rank-based RRF with score-based weighting
        rank_rrf = _BM25_WEIGHT / (_RRF_K + rank + 1)
        score_factor = raw_score / max_bm25  # [0, 1] based on relative BM25 quality
        rrf = rank_rrf * score_factor  # strong BM25 matches get full weight, weak ones discounted
        if cid in scored:
            existing = scored[cid][0]
            merged = SearchScores(rrf=existing.rrf + rrf, embedding=existing.embedding, bm25=raw_score)
            scored[cid] = (merged, scored[cid][1])
        else:
            scored[cid] = (SearchScores(rrf=rrf, embedding=0.0, bm25=raw_score), doc)

    decay_map = {}
    if embed_cache is not None:
        doc_ids = list({_doc_field(scored[cid][1], F_ID) or "" for cid in scored})
        doc_ids = [d for d in doc_ids if d]
        if doc_ids:
            meta = embed_cache.get_decay_metadata(doc_ids)
            for did, (ac, la) in meta.items():
                decay_map[did] = _compute_decay(ac, la)

    if decay_map:
        decayed = {}
        for cid, (scores, doc) in scored.items():
            did = _doc_field(doc, F_ID) or ""
            factor = decay_map.get(did, 1.0)
            decayed[cid] = (SearchScores(rrf=scores.rrf * factor, embedding=scores.embedding, bm25=scores.bm25), doc)
        scored = decayed

    ranked = sorted(scored.items(), key=lambda x: x[1][0].rrf, reverse=True)
    doc_counts: dict[str, int] = defaultdict(int)
    hits = []

    for cid, (scores, doc) in ranked:
        if scores.rrf < params.min_rrf_score or scores.bm25 < params.min_bm25_score or scores.embedding < params.min_embedding_score:
            continue
        doc_id = _doc_field(doc, F_ID) or ""
        if doc_counts[doc_id] >= params.max_hits_per_doc:
            continue
        doc_counts[doc_id] += 1
        ca = _doc_json(doc, F_CHUNK_ATTRIBUTES)
        body = ca.get("body", "") if ca else ""
        attrs = _doc_json(doc, F_ATTRIBUTES)
        hits.append(SearchHit(doc_id=doc_id, chunk_id=cid, body=body, scores=scores, attributes=attrs, chunk_attributes=ca))
        if len(hits) >= params.num_to_return:
            break

    return hits


def _graph_expand(hits, query_vec, graph_store, embed_cache, params):
    """Expand results by traversing graph edges from top hits.

    Finds related documents that the initial search missed by walking
    2 hops through the graph and scoring neighbors by cosine similarity
    to the query embedding.
    """
    if not hits:
        return hits

    qv = np.array(query_vec, dtype=np.float32)
    qv_norm = np.linalg.norm(qv)
    if qv_norm < 1e-12:
        return hits
    qv = qv / qv_norm

    existing_ids = {h.doc_id for h in hits}
    seed_ids = [h.doc_id for h in hits[:5]]  # expand from top 5

    # Collect neighbor doc_ids via multi-hop traversal
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

    # Score candidates by exact cosine similarity to query
    vecs = embed_cache.get_many(list(candidate_ids))
    scored_candidates = []
    for doc_id, vec in vecs.items():
        nv = np.array(vec, dtype=np.float32)
        nv_norm = np.linalg.norm(nv)
        if nv_norm < 1e-12:
            continue
        nv = nv / nv_norm
        sim = float(np.dot(qv, nv))
        if sim >= _GRAPH_MIN_SIM:
            scored_candidates.append((doc_id, sim))

    if not scored_candidates:
        return hits

    # Inject high-scoring graph neighbors into results
    # Give them an RRF score based on their cosine similarity relative to worst hit
    min_rrf = hits[-1].scores.rrf if hits else 0.01
    max_rrf = hits[0].scores.rrf if hits else 0.1

    for doc_id, sim in scored_candidates:
        # Scale RRF score: neighbors get a score between min and max based on similarity
        rrf_score = min_rrf + (max_rrf - min_rrf) * max(0, (sim - _GRAPH_MIN_SIM) / (1.0 - _GRAPH_MIN_SIM))
        scores = SearchScores(rrf=rrf_score, embedding=sim, bm25=0.0)
        hits.append(SearchHit(
            doc_id=doc_id, chunk_id=f"{doc_id}-graph", body="",
            scores=scores, attributes=None, chunk_attributes=None,
        ))

    # Re-sort by RRF and truncate
    hits.sort(key=lambda h: h.scores.rrf, reverse=True)
    return hits[:params.num_to_return]
