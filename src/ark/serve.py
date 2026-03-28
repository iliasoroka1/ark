"""Lightweight HTTP server wrapping the Ark engine.

Exposes search, ingest, and graph operations over HTTP so any agent
can access Ark memory via simple POST requests.
"""

from __future__ import annotations

import hashlib
import json
import os
import traceback

from aiohttp import web

from ark.engine.index import Indexer
from ark.engine.search import Searcher
from ark.engine.graph_store import GraphStore
from ark.engine.circuit_breaker import CircuitBreakerEmbedding
from ark.engine.types import IndexDoc, SearchHit, SearchParams, SearchScores, NodeType
from ark.engine.graph import graph_search
from ark.engine.result import Ok


class ArkServer:
    def __init__(self, host: str = "0.0.0.0", port: int = 7070, data_dir: str | None = None):
        self.host = host
        self.port = port
        self.data_dir = data_dir or os.path.join(os.path.expanduser("~"), ".ark")
        self._indexer: Indexer | None = None
        self._searcher: Searcher | None = None
        self._graph_store: GraphStore | None = None

    def _make_embedding(self):
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

    def _ensure_init(self) -> None:
        if self._indexer is not None:
            return

        memory_dir = self.data_dir
        os.makedirs(memory_dir, exist_ok=True)

        embedding = self._make_embedding()
        self._graph_store = GraphStore(os.path.join(memory_dir, "graph.db"))
        self._indexer = Indexer(embedding=embedding, path=memory_dir, graph_store=self._graph_store)
        cb_embedding = CircuitBreakerEmbedding(embedding)
        self._searcher = Searcher(
            schema=self._indexer.schema,
            index=self._indexer.index,
            embedding=cb_embedding,
            embed_cache=self._indexer.embed_cache,
        )

    def create_app(self) -> web.Application:
        self._ensure_init()
        app = web.Application()
        app.router.add_post("/search", self._handle_search)
        app.router.add_post("/ingest", self._handle_ingest)
        app.router.add_post("/ingest-file", self._handle_ingest_file)
        app.router.add_post("/graph-search", self._handle_graph_search)
        app.router.add_get("/health", self._handle_health)
        return app

    async def _handle_search(self, request: web.Request) -> web.Response:
        try:
            body = await request.json()
            query = body.get("query", "")
            limit = body.get("limit", 10)
            use_case = body.get("use_case")
            tags = body.get("tags")

            params = SearchParams(
                num_to_return=limit,
                num_to_score=limit * 3,
                min_rrf_score=0.005,
                max_hits_per_doc=1,
                use_case=use_case,
            )
            corpus = "agent:ark-serve"

            # Initial vector + BM25 search
            match await self._searcher.search(query, corpus=corpus, params=params):
                case Ok(hits) if hits:
                    # Filter by tags if requested
                    if tags:
                        tag_set = set(tags)
                        hits = [h for h in hits if (h.attributes or {}).get("tag") in tag_set]

                    # Graph-enhanced re-ranking
                    seed_ids = [(h.doc_id, h.scores.rrf) for h in hits]
                    match await self._searcher._embedding.embed(query):
                        case Ok(query_vec):
                            l0_lookup = {h.doc_id: (h.attributes or {}).get("l0", "") for h in hits}
                            gr = graph_search(
                                seed_ids=seed_ids,
                                query_vec=query_vec,
                                graph_store=self._graph_store,
                                embed_cache=self._indexer.embed_cache,
                                l0_lookup=l0_lookup,
                                hops=2,
                            )
                            # Merge graph neighbors into results
                            seen = {h.doc_id for h in hits}
                            for n in gr.neighbors:
                                if n.doc_id not in seen:
                                    seen.add(n.doc_id)
                                    # Fetch content from embedding cache
                                    content = n.l0 or ""
                                    hits.append(SearchHit(
                                        doc_id=n.doc_id,
                                        chunk_id=n.doc_id,
                                        body=content,
                                        scores=SearchScores(rrf=n.score * 0.5, embedding=n.score, bm25=0.0),
                                        node_type=None,
                                        attributes=None,
                                        chunk_attributes=None,
                                    ))
                            # Re-sort by RRF score after adding neighbors
                            hits.sort(key=lambda h: h.scores.rrf, reverse=True)
                        case _:
                            pass  # proceed without graph enhancement

                    results = []
                    for h in hits[:limit]:
                        entry: dict = {
                            "id": h.doc_id,
                            "content": h.body,
                            "score": round(h.scores.rrf, 4),
                        }
                        attrs = h.attributes or {}
                        if attrs.get("l0"):
                            entry["title"] = attrs["l0"]
                        if attrs.get("tag"):
                            entry["tags"] = [attrs["tag"]]
                        if attrs.get("created_at"):
                            entry["created_at"] = attrs["created_at"]
                        results.append(entry)

                    return web.json_response({"results": results})
                case Ok(_):
                    return web.json_response({"results": []})
                case err:
                    return web.json_response({"error": f"Search failed: {err}"}, status=500)
        except Exception as e:
            traceback.print_exc()
            return web.json_response({"error": str(e)}, status=500)

    async def _handle_ingest(self, request: web.Request) -> web.Response:
        try:
            body = await request.json()
            content = body.get("content", "")
            if not content:
                return web.json_response({"error": "content is required"}, status=400)

            title = body.get("title")
            tag = body.get("tag")
            metadata = body.get("metadata", {})
            node_type = body.get("node_type", NodeType.TEXT)

            doc_id = hashlib.sha256(content.encode()).hexdigest()[:16]
            corpus = "agent:ark-serve"

            attrs: dict = {"agent_id": "ark-serve"}
            if title:
                attrs["l0"] = title
            if tag:
                attrs["tag"] = tag
            if metadata:
                attrs.update(metadata)

            doc = IndexDoc(
                id=doc_id,
                source_id="ark-serve",
                corpus=corpus,
                body=content,
                node_type=node_type,
                attributes=attrs,
            )
            result = await self._indexer.add(doc)
            self._indexer.commit()
            self._indexer.index.reload()

            if result.is_err():
                return web.json_response({"error": f"Indexing failed: {result}"}, status=500)

            return web.json_response({"id": doc_id})
        except Exception as e:
            traceback.print_exc()
            return web.json_response({"error": str(e)}, status=500)

    async def _handle_ingest_file(self, request: web.Request) -> web.Response:
        try:
            body = await request.json()
            file_path = body.get("file_path", "")
            if not file_path:
                return web.json_response({"error": "file_path is required"}, status=400)

            file_path = os.path.expanduser(file_path)
            if not os.path.isfile(file_path):
                return web.json_response({"error": f"File not found: {file_path}"}, status=400)

            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()

            if not content.strip():
                return web.json_response({"error": "File is empty"}, status=400)

            title = body.get("title") or os.path.basename(file_path)
            tag = body.get("tag")

            doc_id = hashlib.sha256(content.encode()).hexdigest()[:16]
            corpus = "agent:ark-serve"

            attrs: dict = {"agent_id": "ark-serve", "l0": title, "file_path": file_path}
            if tag:
                attrs["tag"] = tag

            doc = IndexDoc(
                id=doc_id,
                source_id="ark-serve",
                corpus=corpus,
                body=content,
                attributes=attrs,
            )
            result = await self._indexer.add(doc)
            self._indexer.commit()
            self._indexer.index.reload()

            if result.is_err():
                return web.json_response({"error": f"Indexing failed: {result}"}, status=500)

            return web.json_response({"id": doc_id})
        except Exception as e:
            traceback.print_exc()
            return web.json_response({"error": str(e)}, status=500)

    async def _handle_graph_search(self, request: web.Request) -> web.Response:
        try:
            body = await request.json()
            query = body.get("query", "")
            if not query:
                return web.json_response({"error": "query is required"}, status=400)

            hops = max(1, min(3, body.get("hops", 2)))
            diverse = body.get("diverse", False)
            edge_types_str = body.get("edge_types", "")
            et = set(edge_types_str.split(",")) if edge_types_str else None
            corpus = "agent:ark-serve"

            params = SearchParams(
                num_to_return=5,
                num_to_score=20,
                min_rrf_score=0.005,
                max_hits_per_doc=1,
            )

            match await self._searcher.search(query, corpus=corpus, params=params):
                case Ok(hits) if hits:
                    seed_ids = [(h.doc_id, h.scores.rrf) for h in hits]
                    match await self._searcher._embedding.embed(query):
                        case Ok(query_vec):
                            pass
                        case _:
                            return web.json_response({"error": "Failed to embed query"}, status=500)

                    l0_lookup = {h.doc_id: (h.attributes or {}).get("l0", "") for h in hits}
                    result = graph_search(
                        seed_ids=seed_ids,
                        query_vec=query_vec,
                        graph_store=self._graph_store,
                        embed_cache=self._indexer.embed_cache,
                        l0_lookup=l0_lookup,
                        hops=hops,
                        diverse=diverse,
                        edge_types=et,
                    )
                    output = {
                        "seeds": [
                            {"id": s.doc_id, "l0": s.l0, "score": round(s.score, 3)}
                            for s in result.seeds
                        ],
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
                    return web.json_response(output)
                case Ok(_):
                    return web.json_response({"seeds": [], "neighbors": []})
                case err:
                    return web.json_response({"error": f"Search failed: {err}"}, status=500)
        except Exception as e:
            traceback.print_exc()
            return web.json_response({"error": str(e)}, status=500)

    async def _handle_health(self, request: web.Request) -> web.Response:
        try:
            node_count = self._graph_store.count() if self._graph_store else 0
            return web.json_response({"status": "ok", "node_count": node_count})
        except Exception as e:
            return web.json_response({"error": str(e)}, status=500)


def run_server(host: str = "0.0.0.0", port: int = 7070, data_dir: str | None = None) -> None:
    server = ArkServer(host, port, data_dir)
    app = server.create_app()
    resolved_dir = server.data_dir
    print(f"Ark server starting on {host}:{port} (data_dir={resolved_dir})")
    web.run_app(app, host=host, port=port)


if __name__ == "__main__":
    run_server()
