"""Local fallback — call memory/rag tools directly via vendored ark.engine.

No tinyclaw dependency. Opens ~/.tinyclaw/memory tantivy index directly.
"""

from __future__ import annotations

import hashlib
import json
import os
import re

import msgspec

_indexer = None
_searcher = None
_graph_store = None
_initialized = False

_SENT_RE = re.compile(r"[^.!?]*[.!?]")


def _extract_l0(text: str, max_len: int = 200) -> str:
    text = text.strip()
    m = _SENT_RE.match(text)
    if m and len(m.group(0)) <= max_len:
        return m.group(0).strip()
    if len(text) <= max_len:
        return text
    cut = text[:max_len].rsplit(" ", 1)[0]
    return cut + "…"


def _ensure_init() -> None:
    global _indexer, _searcher, _graph_store, _initialized
    if _initialized:
        return

    from ark.engine.index import Indexer
    from ark.engine.search import Searcher
    from ark.engine.graph_store import GraphStore
    from ark.engine.circuit_breaker import CircuitBreakerEmbedding

    memory_dir = os.path.join(os.path.expanduser("~"), ".tinyclaw", "memory")

    embedding = _make_embedding()
    _graph_store = GraphStore(os.path.join(memory_dir, "graph.db"))
    _indexer = Indexer(embedding=embedding, path=memory_dir, graph_store=_graph_store)
    cb_embedding = CircuitBreakerEmbedding(embedding)
    _searcher = Searcher(
        schema=_indexer.schema,
        index=_indexer.index,
        embedding=cb_embedding,
        embed_cache=_indexer.embed_cache,
    )
    _initialized = True


def _make_embedding():
    model = os.environ.get("EMBEDDING_MODEL", "")
    dims = int(os.environ.get("EMBEDDING_DIMS", "1024") or "1024")

    if model:
        from ark.engine.embed import CatsuEmbedding
        return CatsuEmbedding(model=model, dims=dims)

    try:
        from ark.engine.embed import FastEmbedProvider
        return FastEmbedProvider()
    except ImportError:
        pass

    raise RuntimeError(
        "No embedding provider available. Set EMBEDDING_MODEL or install fastembed."
    )


async def call_tool(tool_name: str, payload: dict) -> dict:
    """Call a tool directly, bypassing HTTP."""
    if tool_name == "memory":
        return await _handle_memory(payload)
    return {"ok": False, "error": f"local mode doesn't support tool {tool_name!r}"}


async def _handle_memory(payload: dict) -> dict:
    _ensure_init()
    action = payload.get("action", "")

    if action == "search":
        return await _mem_search(payload.get("query", ""))
    elif action == "add":
        return await _mem_add(payload.get("content", ""), payload.get("tag", ""))
    elif action == "get":
        return await _mem_get(payload.get("id", ""))
    elif action == "list":
        return await _mem_list()
    elif action == "graph_search":
        return await _mem_graph_search(payload)
    elif action == "path":
        return _mem_path(payload.get("from_id", ""), payload.get("id", ""))
    elif action == "analyze":
        return _mem_analyze()
    else:
        return {"ok": False, "error": f"Unknown action '{action}'"}


async def _mem_search(query: str) -> dict:
    from ark.engine.result import Ok
    from ark.engine.types import SearchParams

    params = SearchParams(num_to_return=10, num_to_score=30, min_rrf_score=0.005, max_hits_per_doc=1)
    corpus = "agent:ark-local"

    match await _searcher.search(query, corpus=corpus, params=params):
        case Ok(hits) if hits:
            results = []
            for h in hits:
                l0 = (h.attributes or {}).get("l0") or _extract_l0(h.body)
                entry = {"id": h.doc_id, "l0": l0}
                if h.attributes and h.attributes.get("tag"):
                    entry["tag"] = h.attributes["tag"]
                results.append(entry)
            return {"ok": True, "result": results}
        case Ok(_):
            return {"ok": True, "result": "No memories found."}
        case err:
            return {"ok": False, "error": f"Search failed: {err}"}


async def _mem_add(content: str, tag: str) -> dict:
    from ark.engine.types import IndexDoc

    if not content:
        return {"ok": False, "error": "content is required"}

    doc_id = hashlib.sha256(content.encode()).hexdigest()[:16]
    l0 = _extract_l0(content)
    corpus = "agent:ark-local"

    attrs = {"agent_id": "ark-local", "l0": l0}
    if tag:
        attrs["tag"] = tag

    doc = IndexDoc(id=doc_id, source_id="ark-local", corpus=corpus, body=content, attributes=attrs)
    result = await _indexer.add(doc)
    _indexer.commit()

    if result.is_err():
        return {"ok": False, "error": f"Failed to index: {result}"}

    n = result.unwrap()
    return {"ok": True, "result": {"status": "stored", "id": doc_id, "chunks": n, "l0": l0}}


async def _mem_get(doc_id: str) -> dict:
    import tantivy
    from ark.engine.index import F_CHUNK_ATTRIBUTES, F_CHUNK_ID, F_ID

    if not doc_id:
        return {"ok": False, "error": "id is required"}

    _indexer.index.reload()
    searcher = _indexer.index.searcher()
    query = tantivy.Query.term_query(_indexer.schema, F_ID, doc_id)
    hits = searcher.search(query, limit=50).hits

    if not hits:
        return {"ok": False, "error": f"Memory {doc_id!r} not found."}

    chunks = []
    for _score, addr in hits:
        doc = searcher.doc(addr)
        cids = doc.get_all(F_CHUNK_ID)
        if not cids:
            continue
        ca = doc.get_all(F_CHUNK_ATTRIBUTES)
        if ca and isinstance(ca[0], dict):
            body = ca[0].get("body", "")
            chunks.append((str(cids[0]), body))

    if not chunks:
        return {"ok": False, "error": f"Memory {doc_id!r} has no content."}

    chunks.sort(key=lambda x: x[0])
    full_body = "\n\n".join(body for _, body in chunks)
    return {"ok": True, "result": {"id": doc_id, "content": full_body}}


async def _mem_list() -> dict:
    if _graph_store is not None:
        corpus = "agent:ark-local"
        clusters = _graph_store.list_clusters(corpus)
        if clusters:
            result = {"clusters": [], "unclustered": 0}
            for cid, label, size, node_ids in clusters[:8]:
                result["clusters"].append({"label": label, "count": size})
            return {"ok": True, "result": result}
    return await _mem_search("*")


async def _mem_graph_search(payload: dict) -> dict:
    from ark.engine.result import Ok
    from ark.engine.types import SearchParams
    from ark.engine.graph import graph_search

    query = payload.get("query", "")
    hops = max(1, min(3, payload.get("hops", 2)))
    diverse = payload.get("diverse", False)
    edge_types_str = payload.get("edge_types", "")
    et = set(edge_types_str.split(",")) if edge_types_str else None
    corpus = "agent:ark-local"

    params = SearchParams(num_to_return=5, num_to_score=20, min_rrf_score=0.005, max_hits_per_doc=1)

    match await _searcher.search(query, corpus=corpus, params=params):
        case Ok(hits) if hits:
            seed_ids = [(h.doc_id, h.scores.rrf) for h in hits]
            match await _searcher._embedding.embed(query):
                case Ok(query_vec):
                    pass
                case _:
                    return {"ok": False, "error": "Failed to embed query"}

            l0_lookup = {h.doc_id: (h.attributes or {}).get("l0", "") for h in hits}
            result = graph_search(
                seed_ids=seed_ids, query_vec=query_vec, graph_store=_graph_store,
                embed_cache=_indexer.embed_cache, l0_lookup=l0_lookup,
                hops=hops, diverse=diverse, edge_types=et,
            )
            output = {
                "seeds": [{"id": s.doc_id, "l0": s.l0, "score": round(s.score, 3)} for s in result.seeds],
                "neighbors": [{"id": n.doc_id, "l0": n.l0, "relation": n.relation, "score": round(n.score, 3), "hop": n.hop} for n in result.neighbors],
            }
            return {"ok": True, "result": output}
        case Ok(_):
            return {"ok": True, "result": "No memories found."}
        case err:
            return {"ok": False, "error": f"Search failed: {err}"}


def _mem_path(from_id: str, to_id: str) -> dict:
    if not from_id or not to_id:
        return {"ok": False, "error": "from_id and id are required"}
    if _graph_store is None:
        return {"ok": False, "error": "Graph store not initialized"}
    path = _graph_store.shortest_path(from_id, to_id)
    if path is None:
        return {"ok": True, "result": {"path": None, "message": "No connection found"}}
    steps = [{"id": nid, "via": etype} if etype else {"id": nid} for nid, etype in path]
    return {"ok": True, "result": {"path": steps, "hops": len(steps) - 1}}


def _mem_analyze() -> dict:
    if _graph_store is None or _indexer is None or _indexer.embed_cache is None:
        return {"ok": False, "error": "Graph store or embedding cache not initialized"}
    from ark.engine.spectral import full_analysis
    corpus = "agent:ark-local"
    report = full_analysis(_graph_store, _indexer.embed_cache, corpus)
    return {"ok": True, "result": report}
