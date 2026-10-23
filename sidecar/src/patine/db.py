"""SQLite access. One connection per thread, WAL journal, schema versioned by
PRAGMA user_version. The sidecar is the only writer, so this stays simple."""

from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1

SCHEMA = """
CREATE TABLE IF NOT EXISTS samples (
  id               TEXT PRIMARY KEY,
  path             TEXT NOT NULL UNIQUE,
  hash             TEXT NOT NULL,
  name             TEXT NOT NULL,
  duration         REAL,
  sample_rate      INTEGER,
  channels         INTEGER,
  added_at         TEXT NOT NULL,
  tuning_cents     REAL,
  detected_bpm     REAL,
  bpm_confidence   REAL,
  downbeats        TEXT,            -- JSON [] seconds, may be empty
  beats            TEXT,            -- JSON [] seconds, whole file
  key_estimate     TEXT,
  key_confidence   REAL,
  metrics          TEXT,            -- JSON TonalMetrics measured over the whole file
  analysis_status  TEXT NOT NULL DEFAULT 'pending',   -- pending | running | done | failed
  analysis_error   TEXT,
  analysis_version INTEGER,
  analysis_progress REAL NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS regions (
  id                     TEXT PRIMARY KEY,
  sample_id              TEXT NOT NULL REFERENCES samples(id) ON DELETE CASCADE,
  position               INTEGER NOT NULL DEFAULT 0,
  start_s                REAL NOT NULL,
  end_s                  REAL NOT NULL,
  chord_root             TEXT,            -- 'C'..'B' with sharps, NULL when no chord
  chord_quality          TEXT,            -- maj min 7 m7 maj7 dim sus2 sus4
  bass_note              TEXT,
  chord_confidence       REAL,
  key_context            TEXT,            -- e.g. 'F# minor'
  beats                  TEXT NOT NULL DEFAULT '[]',  -- JSON absolute seconds inside the file
  first_onset_s          REAL,
  declared_length_beats  REAL,
  busyness               REAL,
  register_centroid_hz   REAL,
  metrics                TEXT,            -- JSON TonalMetrics for the region
  user_overrides         TEXT NOT NULL DEFAULT '{}',  -- JSON UserOverrides
  source                 TEXT NOT NULL DEFAULT 'analysis'  -- analysis | user
);
CREATE INDEX IF NOT EXISTS regions_sample ON regions(sample_id, position);

CREATE TABLE IF NOT EXISTS sets (
  id                 TEXT PRIMARY KEY,
  name               TEXT NOT NULL,
  anchor_region_id   TEXT REFERENCES regions(id) ON DELETE SET NULL,
  session_bpm        REAL,
  bpm_pinned         INTEGER NOT NULL DEFAULT 0,
  tonal_settings     TEXT NOT NULL,       -- JSON TonalSettings
  temporal_settings  TEXT NOT NULL,       -- JSON TemporalSettings
  created_at         TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS set_members (
  set_id     TEXT NOT NULL REFERENCES sets(id) ON DELETE CASCADE,
  region_id  TEXT NOT NULL REFERENCES regions(id) ON DELETE CASCADE,
  position   INTEGER NOT NULL DEFAULT 0,
  PRIMARY KEY (set_id, region_id)
);

CREATE TABLE IF NOT EXISTS progressions (
  id          TEXT PRIMARY KEY,
  set_id      TEXT NOT NULL REFERENCES sets(id) ON DELETE CASCADE,
  name        TEXT NOT NULL,
  key         TEXT,                       -- e.g. 'C minor'
  loop        INTEGER NOT NULL DEFAULT 1,
  created_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS slots (
  id              TEXT PRIMARY KEY,
  progression_id  TEXT NOT NULL REFERENCES progressions(id) ON DELETE CASCADE,
  idx             INTEGER NOT NULL,
  chord_symbol    TEXT NOT NULL,
  roman           TEXT,
  chord_root      TEXT,
  chord_quality   TEXT,
  bass_note       TEXT,
  length_beats    REAL NOT NULL DEFAULT 4,
  region_id       TEXT REFERENCES regions(id) ON DELETE SET NULL,
  transpose_semitones INTEGER NOT NULL DEFAULT 0,
  locked          INTEGER NOT NULL DEFAULT 0,
  pattern         TEXT NOT NULL              -- JSON Pattern
);
CREATE INDEX IF NOT EXISTS slots_progression ON slots(progression_id, idx);

CREATE TABLE IF NOT EXISTS renders (
  cache_key   TEXT PRIMARY KEY,
  region_id   TEXT NOT NULL,
  set_id      TEXT NOT NULL,
  path        TEXT NOT NULL,
  status      TEXT NOT NULL,               -- queued | rendering | ready | failed
  error       TEXT,
  params      TEXT NOT NULL,               -- JSON RenderParams used to build the key
  deltas      TEXT,                        -- JSON ConformDeltas
  created_at  TEXT NOT NULL,
  updated_at  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS renders_set ON renders(set_id, region_id);

CREATE TABLE IF NOT EXISTS jobs (
  id          TEXT PRIMARY KEY,
  kind        TEXT NOT NULL,
  status      TEXT NOT NULL,               -- queued | running | done | failed | cancelled
  progress    REAL NOT NULL DEFAULT 0,
  message     TEXT,
  payload     TEXT NOT NULL,               -- JSON
  result      TEXT,                        -- JSON
  error       TEXT,
  created_at  TEXT NOT NULL,
  updated_at  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS jobs_status ON jobs(status, created_at);
"""


def new_id() -> str:
    return uuid.uuid4().hex[:16]


def now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds")


def dumps(obj: Any) -> str:
    return json.dumps(obj, separators=(",", ":"), allow_nan=False)


def loads(text: str | None, default: Any = None) -> Any:
    if text is None or text == "":
        return default
    return json.loads(text)


class Database:
    """Thread-local sqlite3 connections over one file."""

    def __init__(self, path: Path | str):
        self.path = str(path)
        self._local = threading.local()
        self._init_lock = threading.Lock()
        self._initialised = False

    def connect(self) -> sqlite3.Connection:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(self.path, timeout=30, check_same_thread=False)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA foreign_keys = ON")
            if self.path != ":memory:":
                conn.execute("PRAGMA journal_mode = WAL")
                conn.execute("PRAGMA synchronous = NORMAL")
            self._local.conn = conn
        if not self._initialised:
            with self._init_lock:
                if not self._initialised:
                    self._migrate(conn)
                    self._initialised = True
        return conn

    def _migrate(self, conn: sqlite3.Connection) -> None:
        version = conn.execute("PRAGMA user_version").fetchone()[0]
        if version < 1:
            conn.executescript(SCHEMA)
            conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
            conn.commit()
        # Future migrations go here, each guarded by `if version < N:`.

    @contextmanager
    def tx(self) -> Iterator[sqlite3.Connection]:
        conn = self.connect()
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise

    def close(self) -> None:
        conn = getattr(self._local, "conn", None)
        if conn is not None:
            conn.close()
            self._local.conn = None

    # Small helpers -----------------------------------------------------------

    def one(self, sql: str, params: tuple = ()) -> sqlite3.Row | None:
        return self.connect().execute(sql, params).fetchone()

    def all(self, sql: str, params: tuple = ()) -> list[sqlite3.Row]:
        return self.connect().execute(sql, params).fetchall()
