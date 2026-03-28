"""Read side of the hybrid index. Embedding + BM25 search merged via RRF."""

from __future__ import annotations

import logging
from collections import defaultdict
from datetime import UTC, datetime

import numpy as np
import tantivy

from ark.engine.result import Error, Ok, Result
from ark.engine.embed import Embedding
from ark.engine.embedding_cache import EmbeddingCache
from ark.engine.index import (
    F_ATTRIBUTES, F_CHUNK_ATTRIBUTES, F_CHUNK_ID,
    F_CHUNK_TOKENS, F_CORPUS, F_ID, F_SOURCE_ID,
)
from ark.engine.tokenizer import binarize_embedding, tokenize_text
from ark.engine.types import SearchErr, SearchHit, SearchParams, SearchScores

log = logging.getLogger(__name__)

_RRF_K = 15.0
_EMBED_WEIGHT = 2.0
_BM25_WEIGHT = 0.5
_DECAY_FLOOR = 0.3
_DECAY_HALFLIFE = 365.0
_ACCESS_BOOST_CAP = 1.3
_ACCESS_BOOST_STEP = 0.02
_COSINE_BLEND = 0.25  # weight of full-precision cosine in final score


def _expand_query_tokens(tokens: list[str]) -> list[str]:
    """Expand query tokens with morphological variants (light stemming)."""
    expanded = list(tokens)
    seen = set(tokens)
    for tok in tokens:
        variants = []
        # Plural stripping
        if tok.endswith("ies") and len(tok) > 4:
            variants.append(tok[:-3] + "y")
        elif tok.endswith("ses") or tok.endswith("xes") or tok.endswith("zes"):
            variants.append(tok[:-2])
        elif tok.endswith("shes") or tok.endswith("ches"):
            variants.append(tok[:-2])
        elif tok.endswith("s") and not tok.endswith("ss") and len(tok) > 3:
            variants.append(tok[:-1])
        # -ing removal
        if tok.endswith("ing") and len(tok) > 5:
            variants.append(tok[:-3])
            variants.append(tok[:-3] + "e")
        # -ed removal
        if tok.endswith("ed") and len(tok) > 4:
            variants.append(tok[:-2])
            variants.append(tok[:-1])
            if tok.endswith("ied"):
                variants.append(tok[:-3] + "y")
        # -tion/-ment
        if tok.endswith("tion") and len(tok) > 5:
            variants.append(tok[:-4] + "te")
            variants.append(tok[:-4])
        if tok.endswith("ment") and len(tok) > 5:
            variants.append(tok[:-4])
        # Add singular→plural
        if not tok.endswith("s") and len(tok) > 2:
            variants.append(tok + "s")
        for v in variants:
            if v not in seen:
                seen.add(v)
                expanded.append(v)
    return expanded


class Searcher:
    __slots__ = ("_schema", "_index", "_embedding", "_embed_cache")

    def __init__(
        self,
        schema: tantivy.Schema,
        index: tantivy.Index,
        embedding: Embedding,
        embed_cache: EmbeddingCache | None = None,
    ) -> None:
        self._schema = schema
        self._index = index
        self._embedding = embedding
        self._embed_cache = embed_cache

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

        bm25_tokens = tokenize_text(query)
        if bm25_tokens:
            expanded = _expand_query_tokens(bm25_tokens)
            bm25_query = self._build_bm25_query(expanded)
            if corpus or source_ids:
                bm25_query = self._wrap_filters(bm25_query, corpus, source_ids)
            bm25_results = searcher.search(bm25_query, limit=params.num_to_score)
            bm25_docs = [(score, searcher.doc(addr)) for score, addr in bm25_results.hits]
        else:
            bm25_docs = []

        hits = _rrf_merge(embed_docs, bm25_docs, params, self._embed_cache,
                          query_vec=query_vec)
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


def _rrf_merge(embed_docs, bm25_docs, params, embed_cache=None, query_vec=None):
    scored = {}
    for rank, (raw_score, doc) in enumerate(embed_docs):
        cid = _doc_field(doc, F_CHUNK_ID)
        if not cid:
            continue
        rrf = _EMBED_WEIGHT / (_RRF_K + rank + 1)
        scored[cid] = (SearchScores(rrf=rrf, embedding=raw_score, bm25=0.0), doc)

    for rank, (raw_score, doc) in enumerate(bm25_docs):
        cid = _doc_field(doc, F_CHUNK_ID)
        if not cid:
            continue
        rrf = _BM25_WEIGHT / (_RRF_K + rank + 1)
        if cid in scored:
            existing = scored[cid][0]
            merged = SearchScores(rrf=existing.rrf + rrf, embedding=existing.embedding, bm25=raw_score)
            scored[cid] = (merged, scored[cid][1])
        else:
            scored[cid] = (SearchScores(rrf=rrf, embedding=0.0, bm25=raw_score), doc)

    # Stale entry filtering + cosine reranking
    if embed_cache is not None:
        _filter_stale_and_cosine_rerank(scored, embed_cache, query_vec)

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


def _filter_stale_and_cosine_rerank(scored, embed_cache, query_vec):
    """Penalize stale entries and blend full-precision cosine similarity."""
    doc_id_map = {}
    for cid, (scores, doc) in scored.items():
        did = _doc_field(doc, F_ID) or ""
        if did:
            doc_id_map.setdefault(did, []).append(cid)

    if not doc_id_map:
        return

    # Fetch embedding vectors — missing means stale/orphaned
    vecs = embed_cache.get_many(list(doc_id_map.keys()))

    # Penalize stale entries (no cached embedding)
    for did, cids in doc_id_map.items():
        if did not in vecs:
            for cid in cids:
                if cid in scored:
                    old, doc = scored[cid]
                    scored[cid] = (SearchScores(rrf=old.rrf * 0.1, embedding=old.embedding, bm25=old.bm25), doc)

    # Cosine re-ranking with query vector
    if query_vec is None or not vecs:
        return

    try:
        qv = np.array(query_vec, dtype=np.float32)
        qv_norm = np.linalg.norm(qv)
        if qv_norm < 1e-9:
            return

        for did, vec in vecs.items():
            dv = np.array(vec, dtype=np.float32)
            dv_norm = np.linalg.norm(dv)
            if dv_norm < 1e-9:
                continue
            cosine = float(np.dot(qv, dv) / (qv_norm * dv_norm))
            for cid in doc_id_map.get(did, []):
                if cid in scored:
                    old, doc = scored[cid]
                    blended = old.rrf * (1 - _COSINE_BLEND) + cosine * _COSINE_BLEND
                    scored[cid] = (SearchScores(rrf=blended, embedding=old.embedding, bm25=old.bm25), doc)
    except Exception:
        pass  # graceful fallback if numpy ops fail
