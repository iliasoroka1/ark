"""SQLite sidecar for raw embedding vectors."""

from __future__ import annotations

import sqlite3
import struct
from datetime import UTC, datetime
from pathlib import Path


def _pack(vec: list[float]) -> bytes:
    return struct.pack(f"<{len(vec)}f", *vec)


def _unpack(blob: bytes) -> list[float]:
    n = len(blob) // 4
    return list(struct.unpack(f"<{n}f", blob))


class EmbeddingCache:
    __slots__ = ("_conn",)

    def __init__(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(p), timeout=10)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA busy_timeout=5000")
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS embeddings (
                doc_id        TEXT PRIMARY KEY,
                corpus        TEXT NOT NULL,
                dims          INTEGER NOT NULL,
                vec           BLOB NOT NULL,
                access_count  INTEGER NOT NULL DEFAULT 0,
                last_accessed TEXT
            )
        """)
        self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_embeddings_corpus
            ON embeddings(corpus)
        """)
        # Per-agent access tracking (decay/boost scoped per agent)
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS doc_access (
                doc_id        TEXT NOT NULL,
                agent_id      TEXT NOT NULL,
                access_count  INTEGER NOT NULL DEFAULT 0,
                last_accessed TEXT,
                PRIMARY KEY (doc_id, agent_id)
            )
        """)
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS dream_state (
                corpus        TEXT NOT NULL,
                agent_id      TEXT NOT NULL,
                last_dream_at TEXT,
                last_doc_count INTEGER NOT NULL DEFAULT 0,
                PRIMARY KEY (corpus, agent_id)
            )
        """)
        self._migrate()
        self._conn.commit()

    def _migrate(self) -> None:
        cols = {
            row[1]
            for row in self._conn.execute("PRAGMA table_info(embeddings)").fetchall()
        }
        if "access_count" not in cols:
            self._conn.execute(
                "ALTER TABLE embeddings ADD COLUMN access_count INTEGER NOT NULL DEFAULT 0"
            )
        if "last_accessed" not in cols:
            self._conn.execute("ALTER TABLE embeddings ADD COLUMN last_accessed TEXT")
        if "inserted_at" not in cols:
            self._conn.execute("ALTER TABLE embeddings ADD COLUMN inserted_at TEXT")

    def put(self, doc_id: str, corpus: str, vec: list[float]) -> None:
        now = datetime.now(UTC).isoformat()
        self._conn.execute(
            "INSERT OR REPLACE INTO embeddings (doc_id, corpus, dims, vec, inserted_at) VALUES (?, ?, ?, ?, ?)",
            (doc_id, corpus, len(vec), _pack(vec), now),
        )
        self._conn.commit()

    def get(self, doc_id: str) -> list[float] | None:
        row = self._conn.execute(
            "SELECT vec FROM embeddings WHERE doc_id = ?", (doc_id,)
        ).fetchone()
        return _unpack(row[0]) if row else None

    def get_corpus(self, corpus: str) -> list[tuple[str, list[float]]]:
        rows = self._conn.execute(
            "SELECT doc_id, vec FROM embeddings WHERE corpus = ?", (corpus,)
        ).fetchall()
        return [(doc_id, _unpack(blob)) for doc_id, blob in rows]

    def delete(self, doc_id: str) -> None:
        self._conn.execute("DELETE FROM embeddings WHERE doc_id = ?", (doc_id,))
        self._conn.commit()

    def count(self, corpus: str | None = None) -> int:
        if corpus:
            row = self._conn.execute(
                "SELECT COUNT(*) FROM embeddings WHERE corpus = ?", (corpus,)
            ).fetchone()
        else:
            row = self._conn.execute("SELECT COUNT(*) FROM embeddings").fetchone()
        return row[0] if row else 0

    def touch(self, doc_id: str, agent_id: str = "default") -> None:
        now = datetime.now(UTC).isoformat()
        # Per-agent access tracking
        self._conn.execute(
            """INSERT INTO doc_access (doc_id, agent_id, access_count, last_accessed)
               VALUES (?, ?, 1, ?)
               ON CONFLICT(doc_id, agent_id) DO UPDATE SET
                   access_count = access_count + 1,
                   last_accessed = excluded.last_accessed""",
            (doc_id, agent_id, now),
        )
        # Keep legacy global counter in sync for backward compat
        self._conn.execute(
            "UPDATE embeddings SET access_count = access_count + 1, last_accessed = ? WHERE doc_id = ?",
            (now, doc_id),
        )
        self._conn.commit()

    def get_decay_metadata(
        self, doc_ids: list[str], agent_id: str = "default"
    ) -> dict[str, tuple[int, str | None]]:
        if not doc_ids:
            return {}
        placeholders = ",".join("?" for _ in doc_ids)
        # Per-agent access data; falls back to 0 access for docs the agent hasn't touched
        rows = self._conn.execute(
            f"SELECT doc_id, access_count, last_accessed FROM doc_access WHERE agent_id = ? AND doc_id IN ({placeholders})",
            [agent_id] + doc_ids,
        ).fetchall()
        result = {row[0]: (row[1], row[2]) for row in rows}
        # For docs this agent hasn't accessed, return (0, None) so decay still applies by age
        for doc_id in doc_ids:
            if doc_id not in result:
                result[doc_id] = (0, None)
        return result

    def get_many(self, doc_ids: list[str]) -> dict[str, list[float]]:
        if not doc_ids:
            return {}
        placeholders = ",".join("?" for _ in doc_ids)
        rows = self._conn.execute(
            f"SELECT doc_id, vec FROM embeddings WHERE doc_id IN ({placeholders})",
            doc_ids,
        ).fetchall()
        return {doc_id: _unpack(blob) for doc_id, blob in rows}

    def find_similar(
        self,
        doc_id: str,
        corpus: str,
        threshold: float = 0.8,
        limit: int = 5,
    ) -> list[tuple[str, float]]:
        import numpy as np

        query_row = self._conn.execute(
            "SELECT vec FROM embeddings WHERE doc_id = ?", (doc_id,)
        ).fetchone()
        if not query_row:
            return []

        rows = self._conn.execute(
            "SELECT doc_id, vec FROM embeddings WHERE corpus = ? AND doc_id != ?",
            (corpus, doc_id),
        ).fetchall()
        if not rows:
            return []

        q = np.array(_unpack(query_row[0]), dtype=np.float32)
        q_norm = np.linalg.norm(q)
        if q_norm < 1e-12:
            return []
        q = q / q_norm

        ids = [r[0] for r in rows]
        mat = np.array([_unpack(r[1]) for r in rows], dtype=np.float32)
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        mat = mat / norms

        sims = mat @ q
        mask = sims >= threshold
        if not mask.any():
            return []

        indices = np.where(mask)[0]
        scores = sims[indices]
        top_idx = indices[np.argsort(-scores)][:limit]
        return [(ids[i], float(sims[i])) for i in top_idx]

    def search_by_vector(
        self,
        query_vec: list[float],
        corpus: str,
        limit: int = 50,
        threshold: float = 0.0,
    ) -> list[tuple[str, float]]:
        """Search for similar documents by raw query vector. Returns (doc_id, cosine_sim) pairs."""
        import numpy as np

        rows = self._conn.execute(
            "SELECT doc_id, vec FROM embeddings WHERE corpus = ?",
            (corpus,),
        ).fetchall()
        if not rows:
            return []

        q = np.array(query_vec, dtype=np.float32)
        q_norm = np.linalg.norm(q)
        if q_norm < 1e-12:
            return []
        q = q / q_norm

        ids = [r[0] for r in rows]
        mat = np.array([_unpack(r[1]) for r in rows], dtype=np.float32)
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        mat = mat / norms

        sims = mat @ q
        if threshold > 0:
            mask = sims >= threshold
            if not mask.any():
                return []
            indices = np.where(mask)[0]
            scores = sims[indices]
            top_idx = indices[np.argsort(-scores)][:limit]
        else:
            top_idx = np.argsort(-sims)[:limit]

        return [(ids[i], float(sims[i])) for i in top_idx]

    def max_cosine_similarity(self, vec: list[float], corpus: str) -> float:
        rows = self._conn.execute(
            "SELECT vec FROM embeddings WHERE corpus = ?", (corpus,)
        ).fetchall()
        if not rows:
            return 0.0

        import numpy as np

        q = np.array(vec, dtype=np.float32)
        q_norm = np.linalg.norm(q)
        if q_norm < 1e-12:
            return 0.0
        q = q / q_norm

        mat = np.array([_unpack(row[0]) for row in rows], dtype=np.float32)
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        mat = mat / norms

        sims = mat @ q
        return float(sims.max())

    def find_stale(self, corpus: str, max_age_days: int = 90) -> list[str]:
        rows = self._conn.execute(
            "SELECT doc_id FROM embeddings WHERE corpus = ? AND access_count = 0 AND last_accessed IS NULL",
            (corpus,),
        ).fetchall()
        return [row[0] for row in rows]

    # ── Dream state (incremental dreaming) ──

    def get_dream_state(self, corpus: str, agent_id: str) -> tuple[str | None, int]:
        """Return (last_dream_at, last_doc_count) for this corpus+agent."""
        row = self._conn.execute(
            "SELECT last_dream_at, last_doc_count FROM dream_state WHERE corpus = ? AND agent_id = ?",
            (corpus, agent_id),
        ).fetchone()
        return (row[0], row[1]) if row else (None, 0)

    def set_dream_state(self, corpus: str, agent_id: str, doc_count: int) -> None:
        """Update dream state after a dream cycle."""
        now = datetime.now(UTC).isoformat()
        self._conn.execute(
            """INSERT INTO dream_state (corpus, agent_id, last_dream_at, last_doc_count)
               VALUES (?, ?, ?, ?)
               ON CONFLICT(corpus, agent_id) DO UPDATE SET
                   last_dream_at = excluded.last_dream_at,
                   last_doc_count = excluded.last_doc_count""",
            (corpus, agent_id, now, doc_count),
        )
        self._conn.commit()

    def get_new_doc_ids(self, corpus: str, since: str | None) -> list[str]:
        """Return doc_ids added after `since` timestamp (ISO format).

        If since is None, returns all doc_ids (first dream cycle).
        """
        if since is None:
            rows = self._conn.execute(
                "SELECT doc_id FROM embeddings WHERE corpus = ?", (corpus,)
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT doc_id FROM embeddings WHERE corpus = ? AND inserted_at > ?",
                (corpus, since),
            ).fetchall()
        return [row[0] for row in rows]

    def close(self) -> None:
        self._conn.close()
