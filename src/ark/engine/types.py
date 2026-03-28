from __future__ import annotations

from typing import Any

from msgspec import Struct


class NodeType:
    TEXT = 'text'
    CODE_CHUNK = 'code_chunk'
    DECISION = 'decision'
    FINDING = 'finding'
    TASK_RESULT = 'task_result'
    NOTE = 'note'


class CodeChunkMeta(Struct, frozen=True, gc=False):
    file_path: str
    start_line: int
    end_line: int
    language: str | None = None


class IndexDoc(Struct, frozen=True, gc=False):
    id: str
    source_id: str
    corpus: str
    body: str
    node_type: str = NodeType.TEXT
    attributes: dict[str, Any] | None = None


class SearchScores(Struct, frozen=True, gc=False):
    rrf: float = 0.0
    embedding: float = 0.0
    bm25: float = 0.0


class SearchHit(Struct, frozen=True, gc=False):
    doc_id: str
    chunk_id: str
    body: str
    scores: SearchScores
    node_type: str = NodeType.TEXT
    attributes: dict[str, Any] | None = None
    chunk_attributes: dict[str, Any] | None = None


class SearchParams(Struct, frozen=True, gc=False):
    num_to_return: int = 10
    num_to_score: int = 40
    min_rrf_score: float = 0.0
    min_bm25_score: float = 0.0
    min_embedding_score: float = 0.0
    max_hits_per_doc: int = 2
    use_case: str | None = None
    temporal_decay_days: float = 0.0


class IndexErr(Struct, frozen=True, gc=False):
    code: str
    message: str


class SearchErr(Struct, frozen=True, gc=False):
    code: str
    message: str
