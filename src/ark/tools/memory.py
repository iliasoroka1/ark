"""
Memory tool for Tinyclaw agents — add and search durable facts via tantivy.

Graph-aware: every interaction surfaces edge context (derives_from,
contradicts, related_to). No separate graph_search needed for basic use.

Tiered context (L0/L1/L2):
  L0 — one-line abstract, generated extractively at index time.
  L1 — (reserved for future LLM-generated structured overview)
  L2 — full body text.

Search returns L0 + 1-hop edges by default. The agent loads L2 via ``get``.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

import msgspec
import structlog

from tinyclaw.tools.registry import ToolContext, tool
from tinyclaw.tools.result import ToolResult, error, ok, okv

if TYPE_CHECKING:
    from tinyclaw.memory.graph_store import GraphStore
    from tinyclaw.memory.index import Indexer
    from tinyclaw.memory.search import Searcher

log = structlog.get_logger()

_indexer: Indexer | None = None
_searcher: Searcher | None = None
_graph_store: GraphStore | None = None

# Sentence boundary: period/question/exclamation followed by whitespace or end.
_SENT_RE = re.compile(r"[^.!?]*[.!?]")


def _extract_l0(text: str, max_len: int = 200) -> str:
    """Extract L0 abstract: first sentence, capped at *max_len* chars."""
    text = text.strip()
    m = _SENT_RE.match(text)
    if m and len(m.group(0)) <= max_len:
        return m.group(0).strip()
    # No clean sentence break — truncate at word boundary.
    if len(text) <= max_len:
        return text
    cut = text[:max_len].rsplit(" ", 1)[0]
    return cut + "…"


def init(indexer: Indexer, searcher: Searcher, graph_store: "GraphStore | None" = None) -> None:
    """Wire the memory subsystem into the tool. Call once at startup."""
    global _indexer, _searcher, _graph_store
    _indexer = indexer
    _searcher = searcher
    _graph_store = graph_store


def get_searcher() -> Searcher | None:
    """Return the module-level searcher (used by prompt builder)."""
    return _searcher


def get_indexer() -> Indexer | None:
    """Return the module-level indexer (used by deriver)."""
    return _indexer


@tool(
    name="memory",
    description=(
        "Persist and recall knowledge across sessions.\n\n"
        "ACTIONS:\n"
        "- add: Store a fact. Provide content and optionally a tag.\n"
        "  Returns what the fact connected to in your knowledge graph.\n"
        "- search: Find relevant memories. Returns summaries with\n"
        "  related facts (derives_from, contradicts, related_to).\n"
        "- get: Load full content + knowledge connections for a memory.\n"
        "- list: Show your knowledge areas and recent memories by topic.\n"
        "- graph_search: Deep graph traversal. Provide query and\n"
        "  optionally hops (1-3) and diverse (true for variety).\n"
        "  Use when you need to explore chains of reasoning.\n"
        "- path: Trace connection between two memories. Provide from_id + id.\n"
        "  Returns the shortest reasoning chain linking them.\n"
        "- common: Find shared connections between memories. Provide id\n"
        "  (comma-separated IDs). Shows what multiple facts have in common.\n"
        "- analyze: Spectral analysis of your knowledge graph. Returns:\n"
        "  novel memories (RMT), important memories (PageRank), bridge\n"
        "  memories (betweenness), knowledge boundaries (Fiedler),\n"
        "  diverse hubs (entropy). No params needed.\n\n"
        "Search returns summaries + edges. Use get to load full text (L2).\n"
        "~2200 chars of top memories are auto-injected into your system prompt."
    ),
)
async def memory(
    ctx: ToolContext,
    action: str,
    content: str = "",
    query: str = "",
    tag: str = "",
    id: str = "",
    from_id: str = "",
    hops: int = 2,
    diverse: bool = False,
    edge_types: str = "",
) -> ToolResult:
    if _indexer is None or _searcher is None:
        return error(
            "Memory subsystem not initialized. Memory is unavailable in this session."
        )

    agent_id = str(
        ctx.metadata.get("interstellar_agent_id", f"session:{ctx.session_id}")
    )
    corpus = f"agent:{agent_id}"

    if action == "add":
        if not content:
            return error("content is required for add action")
        session_id = str(ctx.metadata.get("session_id_str", f"s:{ctx.session_id}"))
        return await _add(corpus, agent_id, content, tag, session_id=session_id)

    elif action == "search":
        if not query:
            return error("query is required for search action")
        return await _search(corpus, query)

    elif action == "get":
        if not id:
            return error("id is required for get action")
        return await _get(corpus, id)

    elif action == "list":
        return await _list(corpus)

    elif action == "graph_search":
        if not query:
            return error("query is required for graph_search action")
        et = set(edge_types.split(",")) if edge_types else None
        return await _graph_search(corpus, query, hops=hops, diverse=diverse, edge_types=et)

    elif action == "path":
        if not from_id or not id:
            return error("from_id and id are required for path action")
        et = set(edge_types.split(",")) if edge_types else None
        return _path(from_id, id, edge_types=et)

    elif action == "common":
        if not id:
            return error("id is required for common action (comma-separated IDs)")
        node_ids = [x.strip() for x in id.split(",") if x.strip()]
        if len(node_ids) < 2:
            return error("common requires at least 2 IDs (comma-separated)")
        et = set(edge_types.split(",")) if edge_types else None
        return _common(node_ids, edge_types=et)

    elif action == "analyze":
        return _analyze(corpus)

    else:
        return error(f"Unknown action '{action}'. Available: add, search, get, list, graph_search, path, common, analyze")


async def _add(corpus: str, agent_id: str, content: str, tag: str, session_id: str = "") -> ToolResult:
    from tinyclaw.memory.types import IndexDoc

    import hashlib

    doc_id = hashlib.sha256(content.encode()).hexdigest()[:16]

    l0 = _extract_l0(content)

    attrs: dict = {"agent_id": agent_id, "l0": l0}
    if tag:
        attrs["tag"] = tag
    if session_id:
        attrs["session_id"] = session_id

    doc = IndexDoc(
        id=doc_id,
        source_id=agent_id,
        corpus=corpus,
        body=content,
        attributes=attrs,
    )

    result = await _indexer.add(doc)
    _indexer.commit()

    if result.is_err():
        return error(f"Failed to index memory: {result}")

    n = result.unwrap()
    connected_to: list[dict] = []

    # Write graph edges (related_to, same_tag, co_session)
    if _graph_store is not None and _indexer.embed_cache is not None:
        connected_to = _write_add_edges(doc_id, corpus, tag, session_id)

    resp: dict = {"status": "stored", "id": doc_id, "chunks": n, "l0": l0}
    if connected_to:
        resp["connected_to"] = connected_to

    # Entropy production — how much did this fact disrupt the local graph?
    if _graph_store is not None and connected_to:
        from tinyclaw.memory.spectral import entropy_production
        ep = entropy_production(_graph_store, doc_id, corpus)
        if ep > 0.01:  # only report if meaningful
            resp["disruption"] = round(ep, 3)

    return ok(msgspec.json.encode(resp).decode())


def _write_add_edges(doc_id: str, corpus: str, tag: str, session_id: str) -> list[dict]:
    """Write related_to + same_tag + co_session edges. Returns what connected."""
    edges: list[tuple[str, str, str, str, float]] = []
    connected: list[dict] = []

    # 1. related_to — semantic similarity, zero API calls
    similar = _indexer.embed_cache.find_similar(doc_id, corpus, threshold=0.8, limit=5)
    for other_id, sim in similar:
        edges.append((doc_id, other_id, "related_to", corpus, sim))
        edges.append((other_id, doc_id, "related_to", corpus, sim))  # symmetric
        connected.append({"id": other_id, "edge": "related_to", "weight": round(sim, 2)})

    # 2. same_tag — find other docs with same tag
    if tag:
        _find_same_tag(doc_id, corpus, tag, edges, connected)

    # 3. co_session — find other docs from same session
    if session_id:
        _find_co_session(doc_id, corpus, session_id, edges, connected)

    if edges:
        _graph_store.add_edges_batch(edges)

    return connected


def _find_same_tag(doc_id: str, corpus: str, tag: str, edges: list, connected: list) -> None:
    """Find observations in corpus with the same tag via tantivy query."""
    import tantivy
    from tinyclaw.memory.index import F_ATTRIBUTES, F_CHUNK_ID, F_CORPUS, F_ID

    _indexer._index.reload()
    searcher = _indexer._index.searcher()

    # Query for docs in same corpus (parent docs only — no chunk_id)
    corpus_q = tantivy.Query.term_query(_indexer.schema, F_CORPUS, corpus)
    hits = searcher.search(corpus_q, limit=200).hits

    for _score, addr in hits:
        doc = searcher.doc(addr)
        cids = doc.get_all(F_CHUNK_ID)
        if cids:
            continue  # skip chunks
        ids = doc.get_all(F_ID)
        if not ids or str(ids[0]) == doc_id:
            continue
        attrs_raw = doc.get_all(F_ATTRIBUTES)
        if not attrs_raw or not isinstance(attrs_raw[0], dict):
            continue
        attrs = attrs_raw[0]
        if attrs.get("tag") == tag:
            other_id = str(ids[0])
            edges.append((doc_id, other_id, "same_tag", corpus, 0.6))
            edges.append((other_id, doc_id, "same_tag", corpus, 0.6))
            connected.append({"id": other_id, "edge": "same_tag"})


def _find_co_session(doc_id: str, corpus: str, session_id: str, edges: list, connected: list) -> None:
    """Find observations from the same session via tantivy query."""
    import tantivy
    from tinyclaw.memory.index import F_ATTRIBUTES, F_CHUNK_ID, F_CORPUS, F_ID

    _indexer._index.reload()
    searcher = _indexer._index.searcher()

    corpus_q = tantivy.Query.term_query(_indexer.schema, F_CORPUS, corpus)
    hits = searcher.search(corpus_q, limit=200).hits

    for _score, addr in hits:
        doc = searcher.doc(addr)
        cids = doc.get_all(F_CHUNK_ID)
        if cids:
            continue
        ids = doc.get_all(F_ID)
        if not ids or str(ids[0]) == doc_id:
            continue
        attrs_raw = doc.get_all(F_ATTRIBUTES)
        if not attrs_raw or not isinstance(attrs_raw[0], dict):
            continue
        attrs = attrs_raw[0]
        if attrs.get("session_id") == session_id:
            other_id = str(ids[0])
            edges.append((doc_id, other_id, "co_session", corpus, 0.5))
            edges.append((other_id, doc_id, "co_session", corpus, 0.5))
            connected.append({"id": other_id, "edge": "co_session"})


async def _search(corpus: str, query: str) -> ToolResult:
    from tinyclaw.kungfu import Ok
    from tinyclaw.memory.types import SearchParams

    params = SearchParams(
        num_to_return=10,
        num_to_score=30,
        min_rrf_score=0.005,
        max_hits_per_doc=1,
    )

    match await _searcher.search(query, corpus=corpus, params=params):
        case Ok(hits) if hits:
            results = []
            doc_ids = [h.doc_id for h in hits]

            # 1-hop edge annotations
            edge_annotations: dict[str, list[dict]] = {}
            if _graph_store is not None:
                l0_lookup = _build_l0_lookup(doc_ids, hits)
                # Collect all neighbor IDs for L0 lookup
                from tinyclaw.memory.graph import annotate_edges
                edge_annotations = annotate_edges(doc_ids, _graph_store, l0_lookup)

            for h in hits:
                l0 = (h.attributes or {}).get("l0") or _extract_l0(h.body)
                entry: dict = {"id": h.doc_id, "l0": l0}
                if h.attributes and h.attributes.get("tag"):
                    entry["tag"] = h.attributes["tag"]
                if h.doc_id in edge_annotations:
                    entry["edges"] = edge_annotations[h.doc_id]
                results.append(entry)
            return ok(msgspec.json.encode(results).decode())
        case Ok(_):
            return ok("No memories found.")
        case err:
            return error(f"Search failed: {err}")


def _build_l0_lookup(doc_ids: list[str], hits=None) -> dict[str, str]:
    """Build {doc_id: l0} lookup from search hits + graph neighbors."""
    lookup: dict[str, str] = {}
    if hits:
        for h in hits:
            l0 = (h.attributes or {}).get("l0") or _extract_l0(h.body)
            lookup[h.doc_id] = l0

    # Fetch L0 for any neighbor IDs not in hits by querying tantivy parent docs
    if _graph_store is not None:
        neighbor_ids: set[str] = set()
        for doc_id in doc_ids:
            for other_id, _, _, _, _ in _graph_store.get_all_edges(doc_id):
                if other_id not in lookup:
                    neighbor_ids.add(other_id)

        if neighbor_ids:
            lookup.update(_fetch_l0s(list(neighbor_ids)))

    return lookup


def _fetch_l0s(doc_ids: list[str]) -> dict[str, str]:
    """Fetch L0 abstracts from tantivy parent docs."""
    import tantivy
    from tinyclaw.memory.index import F_ATTRIBUTES, F_CHUNK_ID, F_ID

    result: dict[str, str] = {}
    _indexer._index.reload()
    searcher = _indexer._index.searcher()

    for doc_id in doc_ids:
        query = tantivy.Query.term_query(_indexer.schema, F_ID, doc_id)
        hits = searcher.search(query, limit=5).hits
        for _score, addr in hits:
            doc = searcher.doc(addr)
            if doc.get_all(F_CHUNK_ID):
                continue  # skip chunks
            attrs_raw = doc.get_all(F_ATTRIBUTES)
            if attrs_raw and isinstance(attrs_raw[0], dict):
                result[doc_id] = attrs_raw[0].get("l0", "")
            break

    return result


async def _list(corpus: str) -> ToolResult:
    """Cluster-aware memory listing."""
    if _graph_store is not None:
        clusters = _graph_store.list_clusters(corpus)
        if clusters:
            result: dict = {"clusters": [], "unclustered": 0}
            # Get recent memories per cluster
            for cid, label, size, node_ids in clusters[:8]:  # cap at 8 clusters
                recent_l0s = _fetch_l0s(node_ids[:3])  # 3 most recent per cluster
                recent = [{"id": nid, "l0": l0} for nid, l0 in recent_l0s.items() if l0]
                result["clusters"].append({"label": label, "count": size, "recent": recent})

            # Count unclustered (total - sum of cluster sizes)
            total = _indexer.embed_cache.count(corpus) if _indexer.embed_cache else 0
            clustered = sum(c[2] for c in clusters)
            result["unclustered"] = max(0, total - clustered)

            return ok(msgspec.json.encode(result).decode())

    # Fallback: flat search for recent memories
    return await _search(corpus, "*")


def _path(from_id: str, to_id: str, edge_types: set[str] | None = None) -> ToolResult:
    """Shortest path between two observations via bidirectional BFS."""
    if _graph_store is None:
        return error("Graph store not initialized")

    path = _graph_store.shortest_path(from_id, to_id, edge_types=edge_types)
    if path is None:
        return ok(msgspec.json.encode({"path": None, "message": "No connection found"}).decode())

    # Fetch L0s for all nodes in path
    node_ids = [nid for nid, _ in path]
    l0s = _fetch_l0s(node_ids)

    steps = []
    for nid, etype in path:
        step: dict = {"id": nid, "l0": l0s.get(nid, "")}
        if etype:
            step["via"] = etype
        steps.append(step)

    return ok(msgspec.json.encode({"path": steps, "hops": len(steps) - 1}).decode())


def _common(node_ids: list[str], edge_types: set[str] | None = None) -> ToolResult:
    """Find shared connections between multiple observations."""
    if _graph_store is None:
        return error("Graph store not initialized")

    shared = _graph_store.intersect(node_ids, edge_types=edge_types)
    if not shared:
        return ok(msgspec.json.encode({"common": [], "message": "No shared connections"}).decode())

    l0s = _fetch_l0s(list(shared))
    common = [{"id": nid, "l0": l0s.get(nid, "")} for nid in shared if l0s.get(nid)]

    return ok(msgspec.json.encode({"common": common, "count": len(common)}).decode())


async def _graph_search(
    corpus: str, query: str, hops: int = 2, diverse: bool = False, edge_types: set[str] | None = None
) -> ToolResult:
    """Deep graph traversal — multi-hop beam search."""
    from tinyclaw.kungfu import Ok
    from tinyclaw.memory.graph import graph_search
    from tinyclaw.memory.types import SearchParams

    hops = max(1, min(3, hops))  # clamp 1-3

    # Get seeds via normal search
    params = SearchParams(
        num_to_return=5,
        num_to_score=20,
        min_rrf_score=0.005,
        max_hits_per_doc=1,
    )

    match await _searcher.search(query, corpus=corpus, params=params):
        case Ok(hits) if hits:
            seed_ids = [(h.doc_id, h.scores.rrf) for h in hits]

            # Get query embedding for scoring
            from tinyclaw.kungfu import Ok as OkEmbed
            match await _searcher._embedding.embed(query):
                case OkEmbed(query_vec):
                    pass
                case _:
                    return error("Failed to embed query for graph search")

            # Build L0 lookup
            l0_lookup = _build_l0_lookup([h.doc_id for h in hits], hits)

            if _graph_store is None or _indexer.embed_cache is None:
                return error("Graph store not initialized")

            result = graph_search(
                seed_ids=seed_ids,
                query_vec=query_vec,
                graph_store=_graph_store,
                embed_cache=_indexer.embed_cache,
                l0_lookup=l0_lookup,
                hops=hops,
                diverse=diverse,
                edge_types=edge_types,
            )

            output = {
                "seeds": [{"id": s.doc_id, "l0": s.l0, "score": round(s.score, 3)} for s in result.seeds],
                "neighbors": [
                    {
                        "id": n.doc_id,
                        "l0": n.l0,
                        "relation": n.relation,
                        "score": round(n.score, 3),
                        "hop": n.hop,
                    }
                    for n in result.neighbors
                ],
            }
            return ok(msgspec.json.encode(output).decode())

        case Ok(_):
            return ok("No memories found.")
        case err:
            return error(f"Search failed: {err}")


async def _get(corpus: str, doc_id: str) -> ToolResult:
    """Retrieve full body (L2) + edge context for a memory by doc_id."""
    import tantivy
    from tinyclaw.memory.index import F_CHUNK_ATTRIBUTES, F_CHUNK_ID, F_ID

    _indexer._index.reload()
    searcher = _indexer._index.searcher()
    query = tantivy.Query.term_query(_indexer.schema, F_ID, doc_id)
    hits = searcher.search(query, limit=50).hits

    if not hits:
        return error(f"Memory {doc_id!r} not found.")

    # ── Touch: bump access count (deliberate L2 read = real signal) ──
    if _indexer.embed_cache is not None:
        _indexer.embed_cache.touch(doc_id)

    # Collect chunk bodies in order
    chunks: list[tuple[str, str]] = []
    attrs = None
    for _score, addr in hits:
        doc = searcher.doc(addr)
        cids = doc.get_all(F_CHUNK_ID)
        if not cids:
            raw = doc.get_all("attributes")
            if raw and isinstance(raw[0], dict):
                attrs = raw[0]
            continue
        ca = doc.get_all(F_CHUNK_ATTRIBUTES)
        if ca and isinstance(ca[0], dict):
            body = ca[0].get("body", "")
            chunks.append((str(cids[0]), body))

    if not chunks:
        return error(f"Memory {doc_id!r} has no content chunks.")

    chunks.sort(key=lambda x: x[0])
    full_body = "\n\n".join(body for _, body in chunks)

    tag = (attrs or {}).get("tag", "")
    result: dict = {"id": doc_id, "content": full_body}
    if tag:
        result["tag"] = tag

    # Edge context — show what this fact connects to
    if _graph_store is not None:
        all_edges = _graph_store.get_all_edges(doc_id, current_only=True)
        if all_edges:
            neighbor_ids = [e[0] for e in all_edges]
            l0s = _fetch_l0s(neighbor_ids)
            result["edges"] = [
                {
                    "id": other_id,
                    "l0": l0s.get(other_id, ""),
                    "type": edge_type,
                    "direction": direction,
                    "weight": round(weight, 2),
                }
                for other_id, edge_type, direction, weight, _valid_at in all_edges[:10]
            ]

    return ok(msgspec.json.encode(result).decode())


def _analyze(corpus: str) -> ToolResult:
    """Run full spectral analysis of the knowledge graph."""
    if _graph_store is None or _indexer is None or _indexer.embed_cache is None:
        return error("Graph store or embedding cache not initialized")

    from tinyclaw.memory.spectral import full_analysis

    report = full_analysis(_graph_store, _indexer.embed_cache, corpus)

    # Enrich with L0s for all doc_ids mentioned
    all_ids: set[str] = set()
    for section in ("rmt_novel", "pagerank_top", "bridges", "diverse_hubs"):
        for entry in report.get(section, []):
            all_ids.add(entry["id"])
    for entry in report.get("graph_structure", {}).get("boundary_nodes", []):
        all_ids.add(entry["id"])

    if all_ids:
        l0s = _fetch_l0s(list(all_ids))
        for section in ("rmt_novel", "pagerank_top", "bridges", "diverse_hubs"):
            for entry in report.get(section, []):
                entry["l0"] = l0s.get(entry["id"], "")
        for entry in report.get("graph_structure", {}).get("boundary_nodes", []):
            entry["l0"] = l0s.get(entry["id"], "")
