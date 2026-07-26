"""SQLite migration and connection services. Migrations are append-only."""
from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path


MIGRATIONS: tuple[tuple[int, str], ...] = ((1, """
CREATE TABLE application_status (
  key TEXT PRIMARY KEY, value TEXT NOT NULL, updated_at TEXT NOT NULL
);
CREATE TABLE settings (
  key TEXT PRIMARY KEY, value_json TEXT NOT NULL, updated_at TEXT NOT NULL
);
CREATE TABLE jobs (
  id TEXT PRIMARY KEY, type TEXT NOT NULL,
  status TEXT NOT NULL CHECK(status IN ('queued','running','completed','failed','cancelled')),
  progress INTEGER NOT NULL DEFAULT 0 CHECK(progress BETWEEN 0 AND 100),
  stage TEXT, error TEXT, created_at TEXT NOT NULL, started_at TEXT, finished_at TEXT, updated_at TEXT NOT NULL
);
CREATE TABLE audit_events (
  id TEXT PRIMARY KEY, entity_type TEXT NOT NULL, entity_id TEXT NOT NULL,
  action TEXT NOT NULL, detail_json TEXT NOT NULL DEFAULT '{}', created_at TEXT NOT NULL
);
CREATE TABLE input_devices (
  id TEXT PRIMARY KEY, display_name TEXT NOT NULL, metadata_json TEXT NOT NULL DEFAULT '{}', created_at TEXT NOT NULL
);
CREATE TABLE monitors (
  id TEXT PRIMARY KEY, display_name TEXT NOT NULL, metadata_json TEXT NOT NULL DEFAULT '{}', created_at TEXT NOT NULL
);
CREATE TABLE recording_sessions (
  id TEXT PRIMARY KEY, status TEXT NOT NULL CHECK(status IN ('planned','active','completed','discarded')),
  created_at TEXT NOT NULL, finished_at TEXT
);
CREATE TABLE trials (
  id TEXT PRIMARY KEY, session_id TEXT NOT NULL REFERENCES recording_sessions(id), status TEXT NOT NULL,
  created_at TEXT NOT NULL
);
CREATE TABLE dataset_snapshots (
  id TEXT PRIMARY KEY, status TEXT NOT NULL CHECK(status IN ('draft','ready','archived')), created_at TEXT NOT NULL
);
CREATE TABLE preprocessing_runs (
  id TEXT PRIMARY KEY, status TEXT NOT NULL CHECK(status IN ('queued','running','completed','failed','cancelled')), created_at TEXT NOT NULL
);
CREATE TABLE models (
  id TEXT PRIMARY KEY, name TEXT NOT NULL, status TEXT NOT NULL CHECK(status IN ('draft','ready','archived')),
  is_default INTEGER NOT NULL DEFAULT 0 CHECK(is_default IN (0,1)), created_at TEXT NOT NULL
);
CREATE UNIQUE INDEX one_active_default_model ON models(is_default) WHERE is_default = 1 AND status = 'ready';
CREATE TABLE model_artifacts (
  id TEXT PRIMARY KEY, model_id TEXT NOT NULL REFERENCES models(id), path TEXT NOT NULL, created_at TEXT NOT NULL
);
"""), (2, """
CREATE TABLE collection_session_details (
  session_id TEXT PRIMARY KEY REFERENCES recording_sessions(id) ON DELETE CASCADE,
  display_name TEXT NOT NULL,
  mode TEXT NOT NULL CHECK(mode IN ('standard','balanced_coverage','repeated_condition','manual','validation_only')),
  state TEXT NOT NULL CHECK(state IN ('planned','active','paused','completed','failed','cancelled')),
  planned_trials INTEGER NOT NULL CHECK(planned_trials > 0),
  random_seed INTEGER NOT NULL CHECK(random_seed >= 0),
  config_json TEXT NOT NULL,
  environment_json TEXT NOT NULL,
  failure_reason TEXT,
  started_at TEXT,
  ended_at TEXT,
  updated_at TEXT NOT NULL
);
CREATE TABLE trial_details (
  trial_id TEXT PRIMARY KEY REFERENCES trials(id) ON DELETE CASCADE,
  condition_json TEXT NOT NULL,
  state TEXT NOT NULL CHECK(state IN ('target_visible','tracking','completed','failed','cancelled')),
  target_appeared_ns INTEGER NOT NULL CHECK(target_appeared_ns >= 0),
  first_click_ns INTEGER,
  valid_click_ns INTEGER,
  ended_ns INTEGER,
  end_reason TEXT CHECK(end_reason IN ('valid_click','timeout','cancelled','window_focus_lost','capture_failure','user_paused')),
  clicks_json TEXT NOT NULL DEFAULT '[]',
  CHECK(first_click_ns IS NULL OR first_click_ns >= target_appeared_ns),
  CHECK(valid_click_ns IS NULL OR valid_click_ns >= target_appeared_ns),
  CHECK(ended_ns IS NULL OR ended_ns >= target_appeared_ns)
);
CREATE TABLE raw_event_files (
  id TEXT PRIMARY KEY,
  session_id TEXT NOT NULL REFERENCES recording_sessions(id) ON DELETE CASCADE,
  relative_path TEXT NOT NULL,
  format TEXT NOT NULL CHECK(format = 'parquet'),
  status TEXT NOT NULL CHECK(status IN ('open','complete','failed')),
  event_count INTEGER NOT NULL CHECK(event_count >= 0),
  first_timestamp_ns INTEGER,
  last_timestamp_ns INTEGER,
  qpc_frequency_hz INTEGER NOT NULL CHECK(qpc_frequency_hz > 0),
  byte_count INTEGER NOT NULL CHECK(byte_count >= 0),
  sha256 TEXT,
  created_at TEXT NOT NULL,
  finalized_at TEXT,
  CHECK(first_timestamp_ns IS NULL OR first_timestamp_ns >= 0),
  CHECK(last_timestamp_ns IS NULL OR last_timestamp_ns >= first_timestamp_ns),
  CHECK(sha256 IS NULL OR length(sha256) = 64),
  UNIQUE(session_id, relative_path)
);
CREATE TABLE capture_health_events (
  id TEXT PRIMARY KEY,
  session_id TEXT NOT NULL REFERENCES recording_sessions(id) ON DELETE CASCADE,
  severity TEXT NOT NULL CHECK(severity IN ('info','warning','error')),
  code TEXT NOT NULL,
  occurred_at_ns INTEGER NOT NULL CHECK(occurred_at_ns >= 0),
  detail_json TEXT NOT NULL,
  created_at TEXT NOT NULL
);
CREATE INDEX idx_trial_details_state ON trial_details(state);
CREATE INDEX idx_raw_event_files_session ON raw_event_files(session_id);
CREATE INDEX idx_capture_health_session ON capture_health_events(session_id, occurred_at_ns);
"""), (3, """
CREATE TABLE collection_session_reviews (
  session_id TEXT PRIMARY KEY REFERENCES recording_sessions(id) ON DELETE CASCADE,
  disposition TEXT NOT NULL CHECK(disposition IN ('retained','discarded')),
  reason TEXT NOT NULL DEFAULT '',
  reviewed_at TEXT NOT NULL
);
CREATE TABLE trial_reviews (
  trial_id TEXT PRIMARY KEY REFERENCES trials(id) ON DELETE CASCADE,
  disposition TEXT NOT NULL CHECK(disposition IN ('retained','discarded')),
  reason TEXT NOT NULL DEFAULT '',
  reviewed_at TEXT NOT NULL
);
CREATE INDEX idx_trial_reviews_disposition ON trial_reviews(disposition);
"""), (4, """
ALTER TABLE trial_details ADD COLUMN start_screen_x INTEGER;
ALTER TABLE trial_details ADD COLUMN start_screen_y INTEGER;
CREATE TABLE collection_phase_markers (
  id TEXT PRIMARY KEY,
  session_id TEXT NOT NULL REFERENCES recording_sessions(id) ON DELETE CASCADE,
  trial_id TEXT REFERENCES trials(id) ON DELETE CASCADE,
  phase TEXT NOT NULL CHECK(phase IN ('inter_trial','target_visible','trial_completed','session_completed')),
  timestamp_ns INTEGER NOT NULL CHECK(timestamp_ns >= 0),
  screen_x INTEGER,
  screen_y INTEGER,
  detail_json TEXT NOT NULL,
  created_at TEXT NOT NULL,
  CHECK((screen_x IS NULL AND screen_y IS NULL) OR (screen_x IS NOT NULL AND screen_y IS NOT NULL))
);
CREATE INDEX idx_collection_phase_markers_trial ON collection_phase_markers(trial_id, timestamp_ns);
CREATE INDEX idx_collection_phase_markers_session ON collection_phase_markers(session_id, timestamp_ns);
"""),)


def connect(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, timeout=10, isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA busy_timeout = 5000")
    return conn


def migrate(conn: sqlite3.Connection) -> int:
    conn.execute("CREATE TABLE IF NOT EXISTS schema_migrations (version INTEGER PRIMARY KEY, applied_at TEXT NOT NULL)")
    applied = {row[0] for row in conn.execute("SELECT version FROM schema_migrations")}
    from datetime import UTC, datetime
    for version, sql in MIGRATIONS:
        if version not in applied:
            with conn:
                conn.executescript(sql)
                conn.execute("INSERT INTO schema_migrations(version, applied_at) VALUES (?, ?)", (version, datetime.now(UTC).isoformat()))
    result = conn.execute("PRAGMA integrity_check").fetchone()[0]
    if result != "ok":
        raise RuntimeError(f"SQLite integrity check failed: {result}")
    return max(version for version, _ in MIGRATIONS)


@contextmanager
def database(path: Path) -> Iterator[sqlite3.Connection]:
    conn = connect(path)
    try:
        migrate(conn)
        yield conn
    finally:
        conn.close()
