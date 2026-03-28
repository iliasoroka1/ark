import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path


class SessionStore:
    def __init__(self, data_dir: str | None = None):
        if data_dir is None:
            data_dir = os.environ.get("ARK_HOME", str(Path.home() / ".ark"))
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.data_dir / "sessions.db"
        self._init_db()

    def _init_db(self):
        with self._conn() as conn:
            conn.execute(
                """CREATE TABLE IF NOT EXISTS scratchpad (
                    agent_id TEXT,
                    session_id TEXT,
                    key TEXT,
                    value TEXT,
                    updated_at TEXT,
                    PRIMARY KEY (agent_id, session_id, key)
                )"""
            )
            conn.execute(
                """CREATE TABLE IF NOT EXISTS tasks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    agent_id TEXT,
                    session_id TEXT,
                    title TEXT,
                    status TEXT DEFAULT 'todo',
                    created_at TEXT,
                    updated_at TEXT
                )"""
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_tasks_session ON tasks(agent_id, session_id)"
            )
            conn.execute(
                """CREATE TABLE IF NOT EXISTS history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    agent_id TEXT,
                    session_id TEXT,
                    role TEXT,
                    content TEXT,
                    created_at TEXT
                )"""
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_history_session ON history(agent_id, session_id)"
            )

    def _conn(self) -> sqlite3.Connection:
        return sqlite3.connect(str(self.db_path))

    def _now(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    # -- Scratchpad --

    def scratch_set(self, agent_id: str, session_id: str, key: str, value: str):
        now = self._now()
        with self._conn() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO scratchpad (agent_id, session_id, key, value, updated_at) VALUES (?, ?, ?, ?, ?)",
                (agent_id, session_id, key, value, now),
            )

    def scratch_get(self, agent_id: str, session_id: str, key: str) -> str | None:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT value FROM scratchpad WHERE agent_id=? AND session_id=? AND key=?",
                (agent_id, session_id, key),
            ).fetchone()
            return row[0] if row else None

    def scratch_list(self, agent_id: str, session_id: str) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT key, value, updated_at FROM scratchpad WHERE agent_id=? AND session_id=?",
                (agent_id, session_id),
            ).fetchall()
            return [{"key": r[0], "value": r[1], "updated_at": r[2]} for r in rows]

    def scratch_delete(self, agent_id: str, session_id: str, key: str):
        with self._conn() as conn:
            conn.execute(
                "DELETE FROM scratchpad WHERE agent_id=? AND session_id=? AND key=?",
                (agent_id, session_id, key),
            )

    # -- Tasks --

    def task_add(self, agent_id: str, session_id: str, title: str) -> int:
        now = self._now()
        with self._conn() as conn:
            cursor = conn.execute(
                "INSERT INTO tasks (agent_id, session_id, title, status, created_at, updated_at) VALUES (?, ?, ?, 'todo', ?, ?)",
                (agent_id, session_id, title, now, now),
            )
            return cursor.lastrowid

    def task_list(
        self, agent_id: str, session_id: str, status: str | None = None
    ) -> list[dict]:
        with self._conn() as conn:
            if status:
                rows = conn.execute(
                    "SELECT id, title, status, created_at, updated_at FROM tasks WHERE agent_id=? AND session_id=? AND status=?",
                    (agent_id, session_id, status),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT id, title, status, created_at, updated_at FROM tasks WHERE agent_id=? AND session_id=?",
                    (agent_id, session_id),
                ).fetchall()
            return [
                {
                    "id": r[0],
                    "title": r[1],
                    "status": r[2],
                    "created_at": r[3],
                    "updated_at": r[4],
                }
                for r in rows
            ]

    def task_update(
        self, agent_id: str, session_id: str, task_id: int, status: str
    ):
        now = self._now()
        with self._conn() as conn:
            conn.execute(
                "UPDATE tasks SET status=?, updated_at=? WHERE id=? AND agent_id=? AND session_id=?",
                (status, now, task_id, agent_id, session_id),
            )

    def task_complete(self, agent_id: str, session_id: str, task_id: int):
        self.task_update(agent_id, session_id, task_id, "done")

    # -- History --

    def history_add(self, agent_id: str, session_id: str, role: str, content: str):
        now = self._now()
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO history (agent_id, session_id, role, content, created_at) VALUES (?, ?, ?, ?, ?)",
                (agent_id, session_id, role, content, now),
            )

    def history_list(
        self, agent_id: str, session_id: str, limit: int = 50
    ) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT id, role, content, created_at FROM history WHERE agent_id=? AND session_id=? ORDER BY id DESC LIMIT ?",
                (agent_id, session_id, limit),
            ).fetchall()
            return [
                {"id": r[0], "role": r[1], "content": r[2], "created_at": r[3]}
                for r in rows
            ]

    def history_search(
        self, agent_id: str, query: str, limit: int = 10
    ) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT id, session_id, role, content, created_at FROM history WHERE agent_id=? AND content LIKE ? ORDER BY id DESC LIMIT ?",
                (agent_id, f"%{query}%", limit),
            ).fetchall()
            return [
                {
                    "id": r[0],
                    "session_id": r[1],
                    "role": r[2],
                    "content": r[3],
                    "created_at": r[4],
                }
                for r in rows
            ]
