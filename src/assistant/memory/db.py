"""SQLite connection helper for data/brain.db.

Slice 2 adds the people table; slice 5 appends its memory tables to _SCHEMA.
sqlite3 connections are thread-bound: every thread (perception worker, enroll
script, night job) opens its own via connect().
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from assistant.paths import data_dir

DB_FILENAME = "brain.db"

_SCHEMA = [
    """
    CREATE TABLE IF NOT EXISTS people (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        name        TEXT NOT NULL UNIQUE,
        embedding   BLOB NOT NULL,
        enrolled_at REAL NOT NULL,
        last_seen   REAL
    )
    """,
]


def db_path() -> Path:
    return data_dir() / DB_FILENAME


def connect(path: Path | None = None) -> sqlite3.Connection:
    """Open the brain database (creating it and any missing tables) and return the connection."""
    target = path or db_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(target))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    for statement in _SCHEMA:
        conn.execute(statement)
    conn.commit()
    return conn
