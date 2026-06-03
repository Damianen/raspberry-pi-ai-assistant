"""SQLite persistence for scheduled events (alarms, reminders, timers).

Why this exists: a scheduled event must survive a crash or power loss. The
running process is NOT the source of truth — the database is. On startup the
scheduler reloads everything from here, so a reboot never loses an alarm.

(Hardening step for later: fire via systemd timers instead of an in-process
thread, so the alarm fires even if this app isn't running. The DB schema here
already supports that migration.)
"""
from __future__ import annotations

import sqlite3
import threading
from dataclasses import dataclass
from datetime import datetime


@dataclass
class Event:
    id: int
    kind: str            # "alarm" | "reminder" | "timer"
    fire_at: datetime
    label: str
    fired: bool


class Store:
    def __init__(self, path: str = "assistant.db") -> None:
        # check_same_thread=False + our own lock = safe across UI/worker/scheduler
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        self._init()

    def _init(self) -> None:
        with self._lock:
            self._conn.execute(
                """CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    kind TEXT NOT NULL,
                    fire_at TEXT NOT NULL,
                    label TEXT NOT NULL DEFAULT '',
                    fired INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL
                )"""
            )
            self._conn.commit()

    def add(self, kind: str, fire_at: datetime, label: str = "") -> int:
        with self._lock:
            cur = self._conn.execute(
                "INSERT INTO events (kind, fire_at, label, created_at) VALUES (?,?,?,?)",
                (kind, fire_at.isoformat(), label, datetime.now().isoformat()),
            )
            self._conn.commit()
            return int(cur.lastrowid)

    def due(self, now: datetime | None = None) -> list[Event]:
        now = now or datetime.now()
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM events WHERE fired=0 AND fire_at<=? ORDER BY fire_at",
                (now.isoformat(),),
            ).fetchall()
        return [self._row(r) for r in rows]

    def active(self) -> list[Event]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM events WHERE fired=0 ORDER BY fire_at"
            ).fetchall()
        return [self._row(r) for r in rows]

    def mark_fired(self, event_id: int) -> None:
        with self._lock:
            self._conn.execute("UPDATE events SET fired=1 WHERE id=?", (event_id,))
            self._conn.commit()

    @staticmethod
    def _row(r: sqlite3.Row) -> Event:
        return Event(
            id=r["id"], kind=r["kind"],
            fire_at=datetime.fromisoformat(r["fire_at"]),
            label=r["label"], fired=bool(r["fired"]),
        )
