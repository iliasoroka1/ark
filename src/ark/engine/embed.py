from __future__ import annotations

import asyncio
from typing import Protocol, runtime_checkable

import aiohttp
import msgspec

from ark.engine.result import Error, Ok, Result
from ark.engine.types import IndexErr


@runtime_checkable
class Embedding(Protocol):
    @property
    def dims(self) -> int: ...
    async def embed(self, text: str) -> Result[list[float], IndexErr]: ...


async def embed_batch(
    provider: Embedding,
    texts: list[str],
    *,
    batch_size: int = 64,
) -> list[Result[list[float], IndexErr]]:
    if not texts:
        return []
    batches: list[list[tuple[int, str]]] = []
    for i in range(0, len(texts), batch_size):
        batches.append(
            [(j, texts[j]) for j in range(i, min(i + batch_size, len(texts)))]
        )

    results: list[Result[list[float], IndexErr]] = [
        Error(IndexErr(code="pending", message="not yet embedded"))
    ] * len(texts)

    async def _run_batch(batch: list[tuple[int, str]]) -> None:
        coros = [provider.embed(text) for _, text in batch]
        batch_results = await asyncio.gather(*coros)
        for (idx, _), result in zip(batch, batch_results):
            results[idx] = result

    await asyncio.gather(*(_run_batch(b) for b in batches))
    return results


class FastEmbedProvider:
    __slots__ = ("_model", "_dims", "_query_prefix", "_doc_prefix")

    def __init__(self, model_name: str = "nomic-ai/nomic-embed-text-v1.5") -> None:
        from fastembed import TextEmbedding

        self._model = TextEmbedding(model_name=model_name)
        # Nomic models use task prefixes for better retrieval
        if "nomic" in model_name.lower():
            self._query_prefix = "search_query: "
            self._doc_prefix = "search_document: "
        else:
            self._query_prefix = ""
            self._doc_prefix = ""
        probe = list(self._model.embed([self._doc_prefix + "probe"]))[0]
        self._dims = len(probe)

    @property
    def dims(self) -> int:
        return self._dims

    async def embed(self, text: str) -> Result[list[float], IndexErr]:
        """Embed text as a query (uses query prefix for retrieval models)."""
        try:
            prefixed = self._query_prefix + text
            vecs = await asyncio.to_thread(lambda: list(self._model.embed([prefixed])))
            return Ok(vecs[0].tolist())
        except Exception as e:
            return Error(IndexErr(code="embed_error", message=str(e)))

    async def embed_document(self, text: str) -> Result[list[float], IndexErr]:
        """Embed text as a document (uses document prefix for retrieval models)."""
        try:
            prefixed = self._doc_prefix + text
            vecs = await asyncio.to_thread(lambda: list(self._model.embed([prefixed])))
            return Ok(vecs[0].tolist())
        except Exception as e:
            return Error(IndexErr(code="embed_error", message=str(e)))


class _EmbeddingData(msgspec.Struct, frozen=True, gc=False):
    embedding: list[float]


class _EmbeddingResponse(msgspec.Struct, frozen=True, gc=False):
    data: list[_EmbeddingData]


class _EmbeddingError(msgspec.Struct, frozen=True, gc=False):
    message: str
    code: int = 0


class _EmbeddingErrorWrapper(msgspec.Struct, frozen=True, gc=False):
    error: _EmbeddingError


_dec_response = msgspec.json.Decoder(_EmbeddingResponse)
_dec_error = msgspec.json.Decoder(_EmbeddingErrorWrapper)


class CatsuEmbedding:
    __slots__ = ("_client", "_model", "_dims")

    def __init__(self, model: str, dims: int = 0) -> None:
        from catsu import Client

        self._client = Client()
        self._model = model
        self._dims = dims

    @property
    def dims(self) -> int:
        return self._dims

    async def embed(self, text: str) -> Result[list[float], IndexErr]:
        try:
            response = await self._client.aembed(self._model, text)
            if not response.embeddings:
                return Error(IndexErr(code="empty", message="No embeddings returned"))
            vec = response.embeddings[0]
            if self._dims and len(vec) != self._dims:
                return Error(IndexErr(code="dims", message=f"Expected {self._dims} dims, got {len(vec)}"))
            return Ok(vec)
        except Exception as e:
            return Error(IndexErr(code="embed_error", message=str(e)))
