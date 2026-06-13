"""Enrolled-people storage: name -> L2-normalized SFace embedding (float32 blob)."""

from __future__ import annotations

import sqlite3
import time

import numpy as np

EMBEDDING_DTYPE = np.float32


class PeopleStore:
    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    def upsert(self, name: str, embedding: np.ndarray) -> None:
        """Insert or re-enroll a person; a fresh embedding also refreshes enrolled_at."""
        blob = np.asarray(embedding, dtype=EMBEDDING_DTYPE).ravel().tobytes()
        self._conn.execute(
            """
            INSERT INTO people (name, embedding, enrolled_at) VALUES (?, ?, ?)
            ON CONFLICT(name) DO UPDATE SET
                embedding = excluded.embedding,
                enrolled_at = excluded.enrolled_at
            """,
            (name, blob, time.time()),
        )
        self._conn.commit()

    def all_embeddings(self) -> list[tuple[str, np.ndarray]]:
        rows = self._conn.execute("SELECT name, embedding FROM people ORDER BY name").fetchall()
        return [
            (row["name"], np.frombuffer(row["embedding"], dtype=EMBEDDING_DTYPE).copy())
            for row in rows
        ]

    def touch_last_seen(self, name: str) -> None:
        self._conn.execute(
            "UPDATE people SET last_seen = ? WHERE name = ?",
            (time.time(), name),
        )
        self._conn.commit()
