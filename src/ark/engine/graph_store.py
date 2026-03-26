"""SQLite-backed temporal edge store for the observation graph."""

from __future__ import annotations

import json
import sqlite3
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path


class GraphStore:
    __slots__ = ("_conn",)

    def __init__(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(p), timeout=10)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA busy_timeout=5000")
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS edges (
                from_id    TEXT NOT NULL,
                to_id      TEXT NOT NULL,
                edge_type  TEXT NOT NULL,
                corpus     TEXT NOT NULL,
                weight     REAL DEFAULT 1.0,
                valid_at   TEXT NOT NULL,
                invalid_at TEXT,
                PRIMARY KEY (from_id, to_id, edge_type)
            );
            CREATE INDEX IF NOT EXISTS idx_edges_from
                ON edges(from_id, edge_type, invalid_at);
            CREATE INDEX IF NOT EXISTS idx_edges_to
                ON edges(to_id, edge_type, invalid_at);
            CREATE INDEX IF NOT EXISTS idx_edges_corpus
                ON edges(corpus, invalid_at);
            CREATE TABLE IF NOT EXISTS clusters (
                cluster_id TEXT PRIMARY KEY,
                corpus     TEXT NOT NULL,
                label      TEXT NOT NULL,
                size       INTEGER NOT NULL,
                freshness  TEXT NOT NULL,
                node_ids   TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_clusters_corpus ON clusters(corpus);
        """)
        self._conn.commit()

    def add_edge(self, from_id: str, to_id: str, edge_type: str, corpus: str, weight: float = 1.0, valid_at: str | None = None) -> None:
        now = valid_at or datetime.now(UTC).isoformat()
        self._conn.execute(
            "INSERT OR REPLACE INTO edges (from_id, to_id, edge_type, corpus, weight, valid_at) VALUES (?, ?, ?, ?, ?, ?)",
            (from_id, to_id, edge_type, corpus, weight, now),
        )
        self._conn.commit()

    def add_edges_batch(self, edges: list[tuple[str, str, str, str, float]], valid_at: str | None = None) -> None:
        if not edges:
            return
        now = valid_at or datetime.now(UTC).isoformat()
        self._conn.executemany(
            "INSERT OR REPLACE INTO edges (from_id, to_id, edge_type, corpus, weight, valid_at) VALUES (?, ?, ?, ?, ?, ?)",
            [(f, t, et, c, w, now) for f, t, et, c, w in edges],
        )
        self._conn.commit()

    def get_neighbors(self, doc_id: str, current_only: bool = True, edge_types: set[str] | None = None, limit: int = 20) -> list[tuple[str, str, float]]:
        params: list = [doc_id]
        sql = "SELECT to_id, edge_type, weight FROM edges WHERE from_id = ?"
        if edge_types:
            placeholders = ",".join("?" for _ in edge_types)
            sql += f" AND edge_type IN ({placeholders})"
            params.extend(edge_types)
        if current_only:
            sql += " AND invalid_at IS NULL"
        sql += " ORDER BY weight DESC LIMIT ?"
        params.append(limit)
        return self._conn.execute(sql, params).fetchall()

    def get_predecessors(self, doc_id: str, current_only: bool = True, edge_types: set[str] | None = None, limit: int = 20) -> list[tuple[str, str, float]]:
        params: list = [doc_id]
        sql = "SELECT from_id, edge_type, weight FROM edges WHERE to_id = ?"
        if edge_types:
            placeholders = ",".join("?" for _ in edge_types)
            sql += f" AND edge_type IN ({placeholders})"
            params.extend(edge_types)
        if current_only:
            sql += " AND invalid_at IS NULL"
        sql += " ORDER BY weight DESC LIMIT ?"
        params.append(limit)
        return self._conn.execute(sql, params).fetchall()

    def get_all_edges(self, doc_id: str, current_only: bool = True, edge_types: set[str] | None = None) -> list[tuple[str, str, str, float, str]]:
        params_fwd: list = [doc_id]
        params_bwd: list = [doc_id]
        type_filter = ""
        if edge_types:
            placeholders = ",".join("?" for _ in edge_types)
            type_filter = f" AND edge_type IN ({placeholders})"
            type_params = list(edge_types)
            params_fwd.extend(type_params)
            params_bwd.extend(type_params)
        filt = "AND invalid_at IS NULL" if current_only else ""
        sql = f"""
            SELECT to_id, edge_type, 'out', weight, valid_at
            FROM edges WHERE from_id = ? {type_filter} {filt}
            UNION ALL
            SELECT from_id, edge_type, 'in', weight, valid_at
            FROM edges WHERE to_id = ? {type_filter} {filt}
            ORDER BY weight DESC
        """
        return self._conn.execute(sql, params_fwd + params_bwd).fetchall()

    def get_edges_by_type(self, corpus: str, edge_type: str, current_only: bool = True) -> list[tuple[str, str, float]]:
        sql = "SELECT from_id, to_id, weight FROM edges WHERE corpus = ? AND edge_type = ?"
        if current_only:
            sql += " AND invalid_at IS NULL"
        return self._conn.execute(sql, (corpus, edge_type)).fetchall()

    def shortest_path(self, from_id: str, to_id: str, edge_types: set[str] | None = None, current_only: bool = True, max_depth: int = 10) -> list[tuple[str, str | None]] | None:
        if from_id == to_id:
            return [(from_id, None)]
        fwd_parent: dict[str, tuple[str, str]] = {}
        bwd_parent: dict[str, tuple[str, str]] = {}
        fwd_frontier: set[str] = {from_id}
        bwd_frontier: set[str] = {to_id}
        fwd_visited: set[str] = {from_id}
        bwd_visited: set[str] = {to_id}

        def _expand(node: str) -> list[tuple[str, str]]:
            out = [(to, etype) for to, etype, _ in self.get_neighbors(node, current_only=current_only, edge_types=edge_types, limit=100)]
            out += [(fr, etype) for fr, etype, _ in self.get_predecessors(node, current_only=current_only, edge_types=edge_types, limit=100)]
            return out

        for _ in range(max_depth):
            if not fwd_frontier and not bwd_frontier:
                break
            if len(fwd_frontier) <= len(bwd_frontier):
                nxt: set[str] = set()
                for node in fwd_frontier:
                    for neighbor, etype in _expand(node):
                        if neighbor in fwd_visited:
                            continue
                        fwd_parent[neighbor] = (node, etype)
                        fwd_visited.add(neighbor)
                        nxt.add(neighbor)
                        if neighbor in bwd_visited:
                            return self._reconstruct_path(from_id, to_id, neighbor, fwd_parent, bwd_parent)
                fwd_frontier = nxt
            else:
                nxt = set()
                for node in bwd_frontier:
                    for neighbor, etype in _expand(node):
                        if neighbor in bwd_visited:
                            continue
                        bwd_parent[neighbor] = (node, etype)
                        bwd_visited.add(neighbor)
                        nxt.add(neighbor)
                        if neighbor in fwd_visited:
                            return self._reconstruct_path(from_id, to_id, neighbor, fwd_parent, bwd_parent)
                bwd_frontier = nxt
        return None

    def _reconstruct_path(self, from_id, to_id, meeting, fwd_parent, bwd_parent):
        fwd_path: list[tuple[str, str | None]] = []
        node = meeting
        while node != from_id:
            parent_node, etype = fwd_parent[node]
            fwd_path.append((node, etype))
            node = parent_node
        fwd_path.append((from_id, None))
        fwd_path.reverse()
        bwd_path: list[tuple[str, str | None]] = []
        node = meeting
        while node != to_id:
            child_node, etype = bwd_parent[node]
            bwd_path.append((child_node, etype))
            node = child_node
        return fwd_path + bwd_path

    def intersect(self, node_ids: list[str], edge_types: set[str] | None = None, current_only: bool = True) -> set[str]:
        if not node_ids:
            return set()
        result: set[str] | None = None
        for nid in node_ids:
            neighbors = set()
            for to_id, _, _ in self.get_neighbors(nid, current_only=current_only, edge_types=edge_types, limit=1000):
                neighbors.add(to_id)
            for fr_id, _, _ in self.get_predecessors(nid, current_only=current_only, edge_types=edge_types, limit=1000):
                neighbors.add(fr_id)
            if result is None:
                result = neighbors
            else:
                result &= neighbors
                if not result:
                    return set()
        return result or set()

    def count(self, corpus: str | None = None, current_only: bool = True) -> int:
        sql = "SELECT COUNT(*) FROM edges"
        params: list[str] = []
        clauses = []
        if corpus:
            clauses.append("corpus = ?")
            params.append(corpus)
        if current_only:
            clauses.append("invalid_at IS NULL")
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        return self._conn.execute(sql, params).fetchone()[0]

    def is_empty(self, corpus: str) -> bool:
        return self._conn.execute("SELECT 1 FROM edges WHERE corpus = ? LIMIT 1", (corpus,)).fetchone() is None

    def list_clusters(self, corpus: str) -> list[tuple[str, str, int, list[str]]]:
        rows = self._conn.execute(
            "SELECT cluster_id, label, size, node_ids FROM clusters WHERE corpus = ? ORDER BY size DESC",
            (corpus,),
        ).fetchall()
        return [(cid, label, size, json.loads(nids)) for cid, label, size, nids in rows]

    def close(self) -> None:
        self._conn.close()
