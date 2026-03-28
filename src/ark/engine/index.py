"""Write side of the hybrid index. Add/delete documents, commit."""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from pathlib import Path

import tantivy

from ark.engine.result import Error, Ok, Result
from ark.engine.embed import Embedding, embed_batch
from ark.engine.embedding_cache import EmbeddingCache
from ark.engine.tokenizer import Chunker, TextChunker, binarize_embedding, tokenize_text
from ark.engine.types import IndexDoc, IndexErr

log = logging.getLogger(__name__)

DEDUP_THRESHOLD = 0.95

F_CORPUS = "corpus"
F_SOURCE_ID = "source_id"
F_ID = "id"
F_UPDATED_AT = "updated_at"
F_CHUNK_ID = "chunk_id"
F_ATTRIBUTES = "attributes"
F_CHUNK_ATTRIBUTES = "chunk_attributes"
F_CHUNK_TOKENS = "chunk_tokens"
F_FAILED_CHUNKS = "failed_chunks_count"
F_CONTENT_HASH = "content_hash"

_FNV_OFFSET = 0xCBF29CE484222325
_FNV_PRIME = 0x100000001B3
_FNV_MASK = (1 << 64) - 1


def _content_hash(body: str) -> int:
    h = _FNV_OFFSET
    for b in body.encode():
        h = ((h ^ b) * _FNV_PRIME) & _FNV_MASK
    return h


def build_schema() -> tantivy.Schema:
    b = tantivy.SchemaBuilder()
    b.add_text_field(F_CORPUS, stored=True, fast=True, tokenizer_name="raw")
    b.add_text_field(F_SOURCE_ID, stored=True, fast=True, tokenizer_name="raw")
    b.add_text_field(F_ID, stored=True, tokenizer_name="raw")
    b.add_date_field(F_UPDATED_AT, stored=True, indexed=True)
    b.add_text_field(F_CHUNK_ID, stored=True, fast=True, tokenizer_name="raw")
    b.add_json_field(F_ATTRIBUTES, stored=True, tokenizer_name="raw")
    b.add_json_field(F_CHUNK_ATTRIBUTES, stored=True, tokenizer_name="raw")
    b.add_unsigned_field(F_FAILED_CHUNKS, stored=True, indexed=True, fast=True)
    b.add_unsigned_field(F_CONTENT_HASH, stored=True, indexed=True, fast=True)
    b.add_text_field(F_CHUNK_TOKENS, stored=False, tokenizer_name="raw")
    return b.build()


class Indexer:
    __slots__ = (
        "_schema", "_index", "_writer", "_embedding",
        "_chunker", "_embed_cache", "_graph_store",
    )

    def __init__(
        self,
        embedding: Embedding,
        path: str | Path | None = None,
        chunker: Chunker | None = None,
        graph_store=None,
    ) -> None:
        self._schema = build_schema()
        self._embedding = embedding
        self._chunker = chunker or TextChunker()
        self._graph_store = graph_store
        if path is not None:
            p = Path(path)
            p.mkdir(parents=True, exist_ok=True)
            self._index = tantivy.Index(self._schema, path=str(p))
            self._embed_cache: EmbeddingCache | None = EmbeddingCache(p / "embeddings.db")
        else:
            self._index = tantivy.Index(self._schema)
            self._embed_cache = None
        self._writer = self._index.writer()

    @property
    def schema(self) -> tantivy.Schema:
        return self._schema

    @property
    def index(self) -> tantivy.Index:
        return self._index

    @property
    def embed_cache(self) -> EmbeddingCache | None:
        return self._embed_cache

    @property
    def graph_store(self):
        return self._graph_store

    async def add(self, doc: IndexDoc, chunker: Chunker | None = None) -> Result[int, IndexErr]:
        c = chunker or self._chunker
        chunks = c.chunks(doc.body)
        if not chunks:
            return Ok(0)

        now = datetime.now(UTC)
        n = 0
        failed = 0

        embed_results = await embed_batch(self._embedding, chunks)

        for i, body in enumerate(chunks):
            cid = f"{doc.id}-{i}"
            word_tokens = tokenize_text(body)

            result = embed_results[i]
            embed_tokens = result.map(binarize_embedding).unwrap_or([])
            if result.is_err():
                failed += 1

            if self._embed_cache is not None and result.is_ok():
                raw_vec = result.unwrap()
                # Dedup check — skip if very similar content already exists
                # (exclude self by checking before caching)
                sim = self._embed_cache.max_cosine_similarity(raw_vec, doc.corpus)
                if sim >= DEDUP_THRESHOLD:
                    continue
                # Cache embedding after dedup check passes
                if i == 0:
                    self._embed_cache.put(doc.id, doc.corpus, raw_vec)

            all_tokens = word_tokens + embed_tokens

            chunk_doc = tantivy.Document()
            chunk_doc.add_text(F_CORPUS, doc.corpus)
            chunk_doc.add_text(F_SOURCE_ID, doc.source_id)
            chunk_doc.add_text(F_ID, doc.id)
            chunk_doc.add_date(F_UPDATED_AT, now)
            chunk_doc.add_text(F_CHUNK_ID, cid)
            if doc.attributes:
                chunk_doc.add_json(F_ATTRIBUTES, doc.attributes)
            chunk_doc.add_json(F_CHUNK_ATTRIBUTES, {"body": body})
            for tok in all_tokens:
                chunk_doc.add_text(F_CHUNK_TOKENS, tok)
            self._writer.add_document(chunk_doc)
            n += 1

        parent = tantivy.Document()
        parent.add_text(F_CORPUS, doc.corpus)
        parent.add_text(F_SOURCE_ID, doc.source_id)
        parent.add_text(F_ID, doc.id)
        parent.add_date(F_UPDATED_AT, now)
        if doc.attributes:
            parent.add_json(F_ATTRIBUTES, doc.attributes)
        if failed > 0:
            parent.add_unsigned(F_FAILED_CHUNKS, failed)
        parent.add_unsigned(F_CONTENT_HASH, _content_hash(doc.body))
        self._writer.add_document(parent)

        if self._graph_store is not None and doc.attributes:
            self._write_source_edges(doc)

        return Ok(n)

    def _write_source_edges(self, doc: IndexDoc) -> None:
        attrs = doc.attributes or {}
        source_ids = attrs.get("source_ids", [])
        if not source_ids or self._graph_store is None:
            return
        level = attrs.get("observation_level", "")
        edge_type = "contradicts" if level == "contradiction" else "derives_from"
        weight = 0.85 if edge_type == "contradicts" else 0.9
        edges = [(doc.id, sid, edge_type, doc.corpus, weight) for sid in source_ids]
        self._graph_store.add_edges_batch(edges)

    def delete(self, doc_id: str) -> None:
        self._writer.delete_documents(F_ID, doc_id)
        if self._embed_cache is not None:
            self._embed_cache.delete(doc_id)
        if self._graph_store is not None:
            self._graph_store.delete_node(doc_id)

    def commit(self) -> Result[None, IndexErr]:
        try:
            self._writer.commit()
            self._writer.wait_merging_threads()
            self._writer = self._index.writer()
            self._index.reload()
            return Ok(None)
        except Exception as e:
            return Error(IndexErr(code="commit_error", message=str(e)))
