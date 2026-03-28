"""
RAG tool — ingest documents and retrieve relevant chunks via hybrid search.

Uses tinyclaw's existing memory infrastructure (tantivy BM25 + embeddings, RRF)
with a separate index dedicated to document retrieval.

Supports ingesting raw text, files (.txt, .md, .pdf via text extraction),
and searching across all ingested documents.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import TYPE_CHECKING

import msgspec
import structlog

from tinyclaw.memory.tokenizer import SmartChunker, TextChunker
from tinyclaw.tools.registry import ToolContext, tool
from tinyclaw.tools.result import ToolResult, error, ok, okv

if TYPE_CHECKING:
    from tinyclaw.memory.index import Indexer
    from tinyclaw.memory.search import Searcher

log = structlog.get_logger()

_RAG_DIR = os.path.join(os.path.expanduser("~"), ".tinyclaw", "rag")
_CORPUS = "rag:documents"

_indexer: Indexer | None = None
_searcher: Searcher | None = None


def init(indexer: Indexer, searcher: Searcher) -> None:
    """Wire the RAG subsystem. Call once at startup."""
    global _indexer, _searcher
    _indexer = indexer
    _searcher = searcher


def _doc_id(content: str) -> str:
    """Deterministic doc ID from content hash."""
    return hashlib.sha256(content.encode()).hexdigest()[:16]


_BLOCKED_PATTERNS = frozenset(
    (
        ".ssh",
        ".gnupg",
        ".aws/credentials",
        ".netrc",
        ".env",
        ".docker/config.json",
    )
)


def _read_file(path: str) -> str | None:
    """Read text from a file. Supports .txt, .md, .py, .json, .csv, etc."""
    p = Path(path).expanduser().resolve()
    # Block sensitive paths
    parts = p.parts
    for pattern in _BLOCKED_PATTERNS:
        if any(pattern in part for part in parts):
            return None
    if not p.exists():
        return None
    if p.stat().st_size > 10 * 1024 * 1024:  # 10MB limit
        return None
    try:
        return p.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return None


@tool(
    name="rag",
    description=(
        "Ingest documents and search them with hybrid retrieval (BM25 + embeddings).\n\n"
        "ACTIONS:\n"
        "- ingest_text: Index raw text. Provide content and a title.\n"
        "- ingest_file: Index a file from disk. Provide file_path.\n"
        "- search: Find relevant chunks. Provide query. Returns top matches with scores.\n"
        "- list: Show all ingested documents.\n"
        "- delete: Remove a document by its id.\n\n"
        "Documents are chunked, embedded, and stored in a hybrid index. "
        "Search uses BM25 + embedding similarity merged via Reciprocal Rank Fusion.\n"
        "Use tags to organize documents by topic or source."
    ),
)
async def rag(
    ctx: ToolContext,
    action: str,
    content: str = "",
    query: str = "",
    file_path: str = "",
    title: str = "",
    tag: str = "",
    id: str = "",
    limit: int = 10,
) -> ToolResult:
    if _indexer is None or _searcher is None:
        return error("RAG subsystem not initialized. Ensure it's enabled at startup.")

    if action == "ingest_text":
        if not content:
            return error("content is required for ingest_text")
        title = title or content[:60].strip()
        return await _ingest(content, title, tag)

    elif action == "ingest_file":
        if not file_path:
            return error("file_path is required for ingest_file")
        text = _read_file(file_path)
        if text is None:
            return error(f"Could not read file: {file_path}")
        if not text.strip():
            return error(f"File is empty: {file_path}")
        title = title or Path(file_path).name
        tag = tag or "file"
        return await _ingest(text, title, tag, path=file_path)

    elif action == "search":
        if not query:
            return error("query is required for search")
        return await _search(query, limit)

    elif action == "list":
        return await _list_docs()

    elif action == "delete":
        if not id:
            return error("id is required for delete")
        return _delete(id)

    else:
        return error(
            f"Unknown action '{action}'. Available: ingest_text, ingest_file, search, list, delete"
        )


async def _ingest(content: str, title: str, tag: str, path: str = "") -> ToolResult:
    from tinyclaw.memory.types import IndexDoc

    doc_id_val = _doc_id(content)

    # Check if already indexed
    if _indexer.is_indexed(doc_id_val):
        return okv(status="already_indexed", id=doc_id_val, title=title)

    chunker = SmartChunker().for_file(path) if path else TextChunker()
    chunks = chunker.chunks(content)
    attrs = {"title": title, "doc_type": "rag"}
    if tag:
        attrs["tag"] = tag
    attrs["chunk_count"] = str(len(chunks))
    attrs["char_count"] = str(len(content))

    # Index each chunk as a separate doc under the same doc_id
    total_chunks = 0
    for i, chunk in enumerate(chunks):
        chunk_doc = IndexDoc(
            id=doc_id_val,
            source_id=f"rag:{doc_id_val}",
            corpus=_CORPUS,
            body=chunk,
            attributes=attrs if i == 0 else {"title": title, "chunk_idx": str(i)},
        )
        result = await _indexer.add(chunk_doc)
        total_chunks += result.unwrap_or(0)

    _indexer.commit()

    return okv(
        status="ingested",
        id=doc_id_val,
        title=title,
        chunks=total_chunks,
        chars=len(content),
    )


async def _search(query: str, limit: int) -> ToolResult:
    from tinyclaw.kungfu import Ok
    from tinyclaw.memory.types import SearchParams

    params = SearchParams(
        num_to_return=min(limit, 20),
        num_to_score=50,
        min_rrf_score=0.003,
        max_hits_per_doc=3,
    )

    match await _searcher.search(query, corpus=_CORPUS, params=params):
        case Ok(hits) if hits:
            results = []
            for h in hits:
                entry = {
                    "doc_id": h.doc_id,
                    "chunk": h.body[:500] if len(h.body) > 500 else h.body,
                    "score": round(h.scores.rrf, 4),
                }
                if h.attributes:
                    if h.attributes.get("title"):
                        entry["title"] = h.attributes["title"]
                    if h.attributes.get("tag"):
                        entry["tag"] = h.attributes["tag"]
                results.append(entry)
            return ok(msgspec.json.encode(results).decode())
        case Ok(_):
            return ok("No matching documents found.")
        case err:
            return error(f"Search failed: {err}")


async def _list_docs() -> ToolResult:
    """List all ingested RAG documents by scanning the index for unique doc IDs."""
    import tantivy
    from tinyclaw.memory.index import F_ID, F_ATTRIBUTES, F_CORPUS, F_CHUNK_ID

    _indexer._index.reload()
    searcher = _indexer._index.searcher()

    # Find all docs in the rag corpus (parent docs only — no chunk_id)
    corpus_query = tantivy.Query.term_query(_indexer.schema, F_CORPUS, _CORPUS)
    hits = searcher.search(corpus_query, limit=500).hits

    seen = set()
    docs = []
    for _score, addr in hits:
        doc = searcher.doc(addr)
        # Skip chunk docs
        if doc.get_all(F_CHUNK_ID):
            continue
        ids = doc.get_all(F_ID)
        if not ids or str(ids[0]) in seen:
            continue
        doc_id = str(ids[0])
        seen.add(doc_id)

        attrs_raw = doc.get_all(F_ATTRIBUTES)
        attrs = attrs_raw[0] if attrs_raw and isinstance(attrs_raw[0], dict) else {}
        docs.append(
            {
                "id": doc_id,
                "title": attrs.get("title", ""),
                "tag": attrs.get("tag", ""),
                "chunks": attrs.get("chunk_count", "?"),
                "chars": attrs.get("char_count", "?"),
            }
        )

    if not docs:
        return ok("No documents ingested yet.")
    return ok(msgspec.json.encode(docs).decode())


def _delete(doc_id: str) -> ToolResult:
    if not _indexer.is_indexed(doc_id):
        return error(f"Document {doc_id!r} not found.")
    _indexer.delete(doc_id)
    _indexer.commit()
    return okv(status="deleted", id=doc_id)
