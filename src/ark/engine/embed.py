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
        # Retrieval models use task prefixes for asymmetric search
        if "nomic" in model_name.lower():
            self._query_prefix = "search_query: "
            self._doc_prefix = "search_document: "
        elif "bge" in model_name.lower():
            self._query_prefix = "Represent this sentence for searching relevant passages: "
            self._doc_prefix = ""
        elif "e5" in model_name.lower():
            self._query_prefix = "query: "
            self._doc_prefix = "passage: "
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


class OpenRouterEmbedding:
    """Embedding via OpenRouter API (supports Perplexity, BGE, etc.).

    Supports batch embedding: pass a list of texts to embed_batch() for
    efficient single-request processing instead of one call per text.
    """
    __slots__ = ("_model", "_dims", "_api_key")

    def __init__(self, model: str, api_key: str, dims: int = 1024) -> None:
        self._model = model
        self._api_key = api_key
        self._dims = dims

    @property
    def dims(self) -> int:
        return self._dims

    async def _call_api(self, input_data) -> Result[list[list[float]], IndexErr]:
        """Call OpenRouter embeddings API. input_data can be str or list[str]."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    "https://openrouter.ai/api/v1/embeddings",
                    headers={
                        "Authorization": f"Bearer {self._api_key}",
                        "Content-Type": "application/json",
                    },
                    json={"model": self._model, "input": input_data},
                    timeout=aiohttp.ClientTimeout(total=60),
                ) as resp:
                    raw = await resp.read()
                    if resp.status != 200:
                        try:
                            err = _dec_error.decode(raw)
                            return Error(IndexErr(code="api_error", message=err.error.message))
                        except Exception:
                            return Error(IndexErr(code="api_error", message=f"HTTP {resp.status}"))
                    parsed = _dec_response.decode(raw)
                    if not parsed.data:
                        return Error(IndexErr(code="empty", message="No embeddings returned"))
                    vecs = [d.embedding for d in parsed.data]
                    if vecs and self._dims != len(vecs[0]):
                        self._dims = len(vecs[0])
                    return Ok(vecs)
        except Exception as e:
            return Error(IndexErr(code="embed_error", message=str(e)))

    async def embed(self, text: str) -> Result[list[float], IndexErr]:
        result = await self._call_api(text)
        match result:
            case Ok(vecs):
                return Ok(vecs[0])
            case Error(err):
                return Error(err)

    async def embed_document(self, text: str) -> Result[list[float], IndexErr]:
        return await self.embed(text)

    async def embed_batch_texts(self, texts: list[str], batch_size: int = 64) -> list[Result[list[float], IndexErr]]:
        """Embed multiple texts in batched API calls. Much faster than one-by-one."""
        results: list[Result[list[float], IndexErr]] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_result = await self._call_api(batch)
            match batch_result:
                case Ok(vecs):
                    results.extend(Ok(v) for v in vecs)
                case Error(err):
                    results.extend(Error(err) for _ in batch)
        return results


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
