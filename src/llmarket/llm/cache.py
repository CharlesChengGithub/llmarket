"""SQLite cache of LLM calls, keyed by request hash and call coordinates.

Modes:
  off      never read or write;
  use      return a cached answer when present, otherwise call and store (default);
  replay   cached answers only; a miss raises CacheMiss (zero cost, exact reproduction);
  refresh  always call and overwrite.

Rerunning an interrupted experiment in 'use' mode replays every finished call at no cost.
:meth:`Cache.export` copies the calls of one experiment into a standalone file that can be
shipped with a replication package and replayed offline.
"""
from __future__ import annotations

import datetime as _dt
import json
import sqlite3
import threading
from pathlib import Path

from .types import CallCoords, ChatRequest, ChatResponse

MODES = ("off", "use", "replay", "refresh")
DEFAULT_PATH = Path(".llmarket") / "cache.sqlite"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS calls (
  key TEXT PRIMARY KEY,
  request_hash TEXT NOT NULL,
  namespace TEXT, condition TEXT, seed INTEGER, session INTEGER, agent INTEGER,
  period INTEGER, attempt INTEGER,
  provider TEXT, model TEXT,
  request TEXT NOT NULL,
  response TEXT NOT NULL,
  cost REAL, currency TEXT, latency REAL, created TEXT
);
CREATE INDEX IF NOT EXISTS calls_namespace ON calls(namespace, condition, session);
"""
_COLUMNS = ("key", "request_hash", "namespace", "condition", "seed", "session", "agent", "period", "attempt",
            "provider", "model", "request", "response", "cost", "currency", "latency", "created")


class Cache:
    def __init__(self, path: str | Path | None = None, mode: str = "use"):
        if mode is False or mode is None:  # YAML 1.1 reads an unquoted `off` as false
            mode = "off"
        if mode not in MODES:
            raise ValueError(f"cache mode must be one of {MODES}")
        self.mode = mode
        self.path = Path(path) if path is not None else DEFAULT_PATH
        self._lock = threading.Lock()
        self._conn = None
        self.hits = 0
        self.writes = 0
        if mode != "off":
            if mode == "replay" and not self.path.exists():
                raise FileNotFoundError(f"replay mode needs an existing cache file: {self.path}")
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(self.path, check_same_thread=False, timeout=60)
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.executescript(_SCHEMA)

    def get(self, request: ChatRequest, coords: CallCoords) -> ChatResponse | None:
        if self._conn is None or self.mode == "refresh":
            return None
        key = coords.key(request.hash())
        with self._lock:
            row = self._conn.execute("SELECT response, latency FROM calls WHERE key = ?", (key,)).fetchone()
        if row is None:
            return None
        self.hits += 1
        return ChatResponse.from_json(row[0], latency=row[1] or 0.0, cached=True, cost=0.0)

    def put(self, request: ChatRequest, coords: CallCoords, response: ChatResponse, currency: str | None) -> None:
        if self._conn is None or self.mode == "replay":
            return
        rh = request.hash()
        c = coords.as_dict()
        row = (coords.key(rh), rh, c["namespace"], c["condition"], c["seed"], c["session"], c["agent"],
               c["period"], c["attempt"], request.provider, request.model,
               json.dumps({"messages": list(request.messages), "params": request.params}, ensure_ascii=False),
               response.to_json(), response.cost, currency, response.latency,
               _dt.datetime.now().astimezone().isoformat(timespec="seconds"))
        with self._lock:
            self._conn.execute(f"INSERT OR REPLACE INTO calls ({','.join(_COLUMNS)}) VALUES "
                               f"({','.join('?' * len(_COLUMNS))})", row)
            self._conn.commit()
        self.writes += 1

    def count(self, namespace: str | None = None) -> int:
        if self._conn is None:
            return 0
        with self._lock:
            if namespace is None:
                return self._conn.execute("SELECT COUNT(*) FROM calls").fetchone()[0]
            return self._conn.execute("SELECT COUNT(*) FROM calls WHERE namespace = ?", (namespace,)).fetchone()[0]

    def totals(self, namespace: str) -> dict:
        """Calls and recorded cost of one experiment across all runs that wrote to this cache."""
        if self._conn is None:
            return {}
        with self._lock:
            rows = self._conn.execute("SELECT currency, COUNT(*), SUM(cost) FROM calls WHERE namespace = ? "
                                      "GROUP BY currency", (namespace,)).fetchall()
        return {str(cur): dict(calls=n, cost=float(c or 0.0)) for cur, n, c in rows}

    def export(self, path: str | Path, namespace: str, condition: str | None = None) -> int:
        """Copy the calls of one experiment (optionally one condition) into a new cache file."""
        if self._conn is None:
            raise RuntimeError("the cache is off")
        target = Path(path)
        if target.exists():
            raise FileExistsError(target)
        target.parent.mkdir(parents=True, exist_ok=True)
        out = sqlite3.connect(target)
        out.executescript(_SCHEMA)
        q = f"SELECT {','.join(_COLUMNS)} FROM calls WHERE namespace = ?"
        args: tuple = (namespace,)
        if condition is not None:
            q += " AND condition = ?"
            args += (condition,)
        with self._lock:
            rows = self._conn.execute(q, args).fetchall()
        out.executemany(f"INSERT INTO calls ({','.join(_COLUMNS)}) VALUES ({','.join('?' * len(_COLUMNS))})", rows)
        out.commit()
        out.close()
        return len(rows)

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None
