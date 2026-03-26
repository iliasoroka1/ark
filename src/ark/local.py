"""Local fallback — call tinyclaw tools directly when the server is down.

Initializes the memory subsystem lazily on first use, reuses it after.
Requires tinyclaw to be importable (installed or on PYTHONPATH).
"""

from __future__ import annotations

import json
import os

_initialized = False


def _ensure_init() -> None:
    """Initialize tinyclaw memory + rag subsystems once."""
    global _initialized
    if _initialized:
        return

    memory_dir = os.path.join(os.path.expanduser("~"), ".tinyclaw", "memory")
    rag_dir = os.path.join(os.path.expanduser("~"), ".tinyclaw", "rag")

    # Memory
    try:
        from tinyclaw.memory.index import Indexer
        from tinyclaw.memory.search import Searcher
        from tinyclaw.memory.graph_store import GraphStore
        from tinyclaw.memory.circuit_breaker import CircuitBreakerEmbedding
        from tinyclaw.tools.memory import init as init_memory_tool

        embedding = _make_embedding()
        graph_store = GraphStore(os.path.join(memory_dir, "graph.db"))
        indexer = Indexer(embedding=embedding, path=memory_dir, graph_store=graph_store)
        cb_embedding = CircuitBreakerEmbedding(embedding)
        searcher = Searcher(
            schema=indexer.schema,
            index=indexer.index,
            embedding=cb_embedding,
            embed_cache=indexer.embed_cache,
        )
        init_memory_tool(indexer, searcher, graph_store=graph_store)
    except Exception as e:
        raise RuntimeError(f"Failed to init memory locally: {e}") from e

    # RAG
    try:
        from tinyclaw.tools.rag import init as init_rag_tool

        rag_embedding = _make_embedding()
        rag_cb = CircuitBreakerEmbedding(rag_embedding)
        rag_indexer = Indexer(embedding=rag_embedding, path=rag_dir)
        rag_searcher = Searcher(
            schema=rag_indexer.schema,
            index=rag_indexer.index,
            embedding=rag_cb,
        )
        init_rag_tool(rag_indexer, rag_searcher)
    except Exception:
        pass  # RAG is optional

    _initialized = True


def _make_embedding():
    """Build embedding provider from env vars, matching tinyclaw's logic."""
    model = os.environ.get("EMBEDDING_MODEL", "")
    dims = int(os.environ.get("EMBEDDING_DIMS", "1024") or "1024")

    if model:
        from tinyclaw.memory.embed import CatsuEmbedding
        return CatsuEmbedding(model=model, dims=dims)

    # Fallback: fastembed (local, no API key needed)
    try:
        from tinyclaw.memory.embed import FastEmbedProvider
        return FastEmbedProvider()
    except ImportError:
        pass

    raise RuntimeError(
        "No embedding provider available. Set EMBEDDING_MODEL or install fastembed."
    )


async def call_tool(tool_name: str, payload: dict) -> dict:
    """Call a tinyclaw tool directly, bypassing HTTP."""
    _ensure_init()

    from tinyclaw.tools.registry import ToolContext

    # Import the tool function by name
    if tool_name == "memory":
        from tinyclaw.tools.memory import memory as tool_fn
    elif tool_name == "rag":
        from tinyclaw.tools.rag import rag as tool_fn
    elif tool_name == "session_history":
        from tinyclaw.tools.session_history import session_history as tool_fn
    else:
        return {"ok": False, "error": f"local mode doesn't support tool {tool_name!r}"}

    ctx = ToolContext(workspace=os.getcwd(), session_id=0, metadata={"source": "ark-local"})

    result = await tool_fn(ctx, **payload)

    if result.is_error:
        return {"ok": False, "error": result.content}

    try:
        parsed = json.loads(result.content)
        return {"ok": True, "result": parsed}
    except (json.JSONDecodeError, TypeError):
        return {"ok": True, "result": result.content}
