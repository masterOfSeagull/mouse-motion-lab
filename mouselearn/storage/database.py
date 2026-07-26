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
"""), (5, """
CREATE TABLE dataset_snapshot_details (
  snapshot_id TEXT PRIMARY KEY REFERENCES dataset_snapshots(id) ON DELETE RESTRICT,
  name TEXT NOT NULL,
  description TEXT NOT NULL DEFAULT '',
  ordered_trial_ids_json TEXT NOT NULL,
  raw_session_hashes_json TEXT NOT NULL,
  preprocessing_config_json TEXT NOT NULL,
  split_config_json TEXT NOT NULL,
  warnings_json TEXT NOT NULL DEFAULT '[]',
  feature_schema_version INTEGER NOT NULL CHECK(feature_schema_version > 0),
  code_revision TEXT NOT NULL,
  manifest_sha256 TEXT,
  manifest_relative_path TEXT,
  trial_count INTEGER NOT NULL CHECK(trial_count >= 0),
  session_count INTEGER NOT NULL CHECK(session_count >= 0),
  CHECK((manifest_sha256 IS NULL AND manifest_relative_path IS NULL) OR
        (length(manifest_sha256) = 64 AND manifest_relative_path IS NOT NULL))
);
CREATE TABLE dataset_snapshot_trials (
  snapshot_id TEXT NOT NULL REFERENCES dataset_snapshots(id) ON DELETE RESTRICT,
  trial_id TEXT NOT NULL REFERENCES trials(id) ON DELETE RESTRICT,
  session_id TEXT NOT NULL REFERENCES recording_sessions(id) ON DELETE RESTRICT,
  split TEXT NOT NULL CHECK(split IN ('train','validation','test')),
  ordinal INTEGER NOT NULL CHECK(ordinal >= 0),
  PRIMARY KEY(snapshot_id, trial_id),
  UNIQUE(snapshot_id, ordinal)
);
CREATE INDEX idx_dataset_snapshot_trials_snapshot_split ON dataset_snapshot_trials(snapshot_id, split, ordinal);
CREATE TRIGGER dataset_snapshot_details_immutable
BEFORE UPDATE ON dataset_snapshot_details
WHEN (SELECT status FROM dataset_snapshots WHERE id=OLD.snapshot_id) = 'ready'
BEGIN SELECT RAISE(ABORT, 'ready dataset snapshots are immutable'); END;
CREATE TRIGGER dataset_snapshot_trials_insert_immutable
BEFORE INSERT ON dataset_snapshot_trials
WHEN (SELECT status FROM dataset_snapshots WHERE id=NEW.snapshot_id) = 'ready'
BEGIN SELECT RAISE(ABORT, 'ready dataset snapshots are immutable'); END;
CREATE TRIGGER dataset_snapshot_trials_update_immutable
BEFORE UPDATE ON dataset_snapshot_trials
WHEN (SELECT status FROM dataset_snapshots WHERE id=OLD.snapshot_id) = 'ready'
BEGIN SELECT RAISE(ABORT, 'ready dataset snapshots are immutable'); END;
CREATE TRIGGER dataset_snapshot_trials_delete_immutable
BEFORE DELETE ON dataset_snapshot_trials
WHEN (SELECT status FROM dataset_snapshots WHERE id=OLD.snapshot_id) = 'ready'
BEGIN SELECT RAISE(ABORT, 'ready dataset snapshots are immutable'); END;
"""), (6, """
CREATE TABLE collection_session_quality (
  session_id TEXT PRIMARY KEY REFERENCES recording_sessions(id) ON DELETE CASCADE,
  classification TEXT NOT NULL CHECK(classification IN ('legacy','current')),
  reasons_json TEXT NOT NULL,
  assessed_at TEXT NOT NULL
);
INSERT INTO collection_session_quality(session_id,classification,reasons_json,assessed_at)
SELECT id, 'legacy', '["scheduler_not_cursor_relative","target_presentation_not_confirmed"]', CURRENT_TIMESTAMP
FROM recording_sessions;
DROP INDEX idx_collection_phase_markers_trial;
DROP INDEX idx_collection_phase_markers_session;
CREATE TABLE collection_phase_markers_next (
  id TEXT PRIMARY KEY,
  session_id TEXT NOT NULL REFERENCES recording_sessions(id) ON DELETE CASCADE,
  trial_id TEXT REFERENCES trials(id) ON DELETE CASCADE,
  phase TEXT NOT NULL CHECK(phase IN (
    'inter_trial','target_visible','trial_completed','trial_cancelled','trial_failed','trial_timed_out',
    'session_completed','session_cancelled','session_failed'
  )),
  timestamp_ns INTEGER NOT NULL CHECK(timestamp_ns >= 0),
  screen_x INTEGER,
  screen_y INTEGER,
  detail_json TEXT NOT NULL,
  created_at TEXT NOT NULL,
  CHECK((screen_x IS NULL AND screen_y IS NULL) OR (screen_x IS NOT NULL AND screen_y IS NOT NULL))
);
INSERT INTO collection_phase_markers_next
SELECT id,session_id,trial_id,phase,timestamp_ns,screen_x,screen_y,detail_json,created_at
FROM collection_phase_markers;
DROP TABLE collection_phase_markers;
ALTER TABLE collection_phase_markers_next RENAME TO collection_phase_markers;
CREATE INDEX idx_collection_phase_markers_trial ON collection_phase_markers(trial_id, timestamp_ns);
CREATE INDEX idx_collection_phase_markers_session ON collection_phase_markers(session_id, timestamp_ns);
"""), (7, """
CREATE TABLE preprocessing_run_details (
  run_id TEXT PRIMARY KEY REFERENCES preprocessing_runs(id) ON DELETE CASCADE,
  snapshot_id TEXT NOT NULL REFERENCES dataset_snapshots(id) ON DELETE RESTRICT,
  config_json TEXT NOT NULL,
  code_revision TEXT NOT NULL,
  processed_relative_path TEXT,
  processed_sha256 TEXT,
  report_relative_path TEXT,
  report_sha256 TEXT,
  processed_trial_count INTEGER NOT NULL DEFAULT 0 CHECK(processed_trial_count >= 0),
  skipped_trial_count INTEGER NOT NULL DEFAULT 0 CHECK(skipped_trial_count >= 0),
  error TEXT,
  started_at TEXT,
  finished_at TEXT,
  CHECK((processed_relative_path IS NULL AND processed_sha256 IS NULL) OR
        (processed_relative_path IS NOT NULL AND length(processed_sha256) = 64)),
  CHECK((report_relative_path IS NULL AND report_sha256 IS NULL) OR
        (report_relative_path IS NOT NULL AND length(report_sha256) = 64))
);
CREATE INDEX idx_preprocessing_run_details_snapshot ON preprocessing_run_details(snapshot_id, run_id);
"""), (8, """
CREATE TABLE model_details (
  model_id TEXT PRIMARY KEY REFERENCES models(id) ON DELETE CASCADE,
  model_type TEXT NOT NULL CHECK(model_type IN ('retrieval','pca_mixture')),
  dataset_snapshot_id TEXT NOT NULL REFERENCES dataset_snapshots(id) ON DELETE RESTRICT,
  preprocessing_run_id TEXT NOT NULL REFERENCES preprocessing_runs(id) ON DELETE RESTRICT,
  config_json TEXT NOT NULL,
  code_revision TEXT NOT NULL,
  manifest_relative_path TEXT,
  manifest_sha256 TEXT,
  validation_relative_path TEXT,
  validation_sha256 TEXT,
  error TEXT,
  CHECK((manifest_relative_path IS NULL AND manifest_sha256 IS NULL) OR
        (manifest_relative_path IS NOT NULL AND length(manifest_sha256) = 64)),
  CHECK((validation_relative_path IS NULL AND validation_sha256 IS NULL) OR
        (validation_relative_path IS NOT NULL AND length(validation_sha256) = 64))
);
CREATE INDEX idx_model_details_preprocessing_run ON model_details(preprocessing_run_id, model_type);
"""), (9, """
ALTER TABLE model_details RENAME TO model_details_v8;
CREATE TABLE model_details (
  model_id TEXT PRIMARY KEY REFERENCES models(id) ON DELETE CASCADE,
  model_type TEXT NOT NULL CHECK(model_type IN ('retrieval','pca_mixture','conditional_flow')),
  dataset_snapshot_id TEXT NOT NULL REFERENCES dataset_snapshots(id) ON DELETE RESTRICT,
  preprocessing_run_id TEXT NOT NULL REFERENCES preprocessing_runs(id) ON DELETE RESTRICT,
  config_json TEXT NOT NULL, code_revision TEXT NOT NULL,
  manifest_relative_path TEXT, manifest_sha256 TEXT,
  validation_relative_path TEXT, validation_sha256 TEXT, error TEXT,
  CHECK((manifest_relative_path IS NULL AND manifest_sha256 IS NULL) OR
        (manifest_relative_path IS NOT NULL AND length(manifest_sha256) = 64)),
  CHECK((validation_relative_path IS NULL AND validation_sha256 IS NULL) OR
        (validation_relative_path IS NOT NULL AND length(validation_sha256) = 64))
);
INSERT INTO model_details SELECT * FROM model_details_v8;
DROP TABLE model_details_v8;
CREATE INDEX idx_model_details_preprocessing_run ON model_details(preprocessing_run_id, model_type);
CREATE TABLE experiments (
  id TEXT PRIMARY KEY,
  name TEXT NOT NULL,
  status TEXT NOT NULL CHECK(status IN ('queued','running','completed','failed','cancelled')),
  dataset_snapshot_id TEXT NOT NULL REFERENCES dataset_snapshots(id) ON DELETE RESTRICT,
  preprocessing_run_id TEXT NOT NULL REFERENCES preprocessing_runs(id) ON DELETE RESTRICT,
  model_id TEXT REFERENCES models(id) ON DELETE SET NULL,
  config_json TEXT NOT NULL,
  random_seed INTEGER NOT NULL CHECK(random_seed >= 0),
  latest_epoch INTEGER NOT NULL DEFAULT 0 CHECK(latest_epoch >= 0),
  latest_metrics_json TEXT NOT NULL DEFAULT '{}',
  best_validation_loss REAL,
  checkpoint_relative_path TEXT,
  error TEXT,
  created_at TEXT NOT NULL, started_at TEXT, finished_at TEXT
);
CREATE INDEX idx_experiments_status_created ON experiments(status, created_at);
"""), (10, """
ALTER TABLE experiments ADD COLUMN job_id TEXT REFERENCES jobs(id) ON DELETE SET NULL;
CREATE INDEX idx_experiments_job ON experiments(job_id);
"""), (11, """
CREATE TABLE model_registry (
  model_id TEXT PRIMARY KEY REFERENCES models(id) ON DELETE CASCADE,
  lifecycle TEXT NOT NULL CHECK(lifecycle IN ('candidate','validated','active','deprecated')),
  validation_sha256 TEXT,
  updated_at TEXT NOT NULL,
  CHECK(validation_sha256 IS NULL OR length(validation_sha256) = 64)
);
CREATE UNIQUE INDEX one_active_registry_model ON model_registry(lifecycle) WHERE lifecycle='active';
INSERT INTO model_registry(model_id,lifecycle,validation_sha256,updated_at)
SELECT m.id,CASE WHEN d.validation_sha256 IS NULL THEN 'candidate' ELSE 'validated' END,d.validation_sha256,CURRENT_TIMESTAMP
FROM models m JOIN model_details d ON d.model_id=m.id WHERE m.status='ready';
"""),)


def connect(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, timeout=10, isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA busy_timeout = 5000")
    return conn


def _execute_script_transactionally(conn: sqlite3.Connection, script: str) -> None:
    """Execute complete SQLite statements without executescript's implicit commits."""
    statement = ""
    for character in script:
        statement += character
        if sqlite3.complete_statement(statement):
            if statement.strip():
                conn.execute(statement)
            statement = ""
    if statement.strip():
        raise RuntimeError("migration contains an incomplete SQL statement")


def migrate(conn: sqlite3.Connection, migrations: tuple[tuple[int, str], ...] = MIGRATIONS) -> int:
    conn.execute("CREATE TABLE IF NOT EXISTS schema_migrations (version INTEGER PRIMARY KEY, applied_at TEXT NOT NULL)")
    applied = {row[0] for row in conn.execute("SELECT version FROM schema_migrations")}
    from datetime import UTC, datetime
    for version, sql in migrations:
        if version not in applied:
            conn.execute("BEGIN IMMEDIATE")
            try:
                _execute_script_transactionally(conn, sql)
                conn.execute("INSERT INTO schema_migrations(version, applied_at) VALUES (?, ?)", (version, datetime.now(UTC).isoformat()))
                conn.execute("COMMIT")
            except Exception:
                conn.execute("ROLLBACK")
                raise
    result = conn.execute("PRAGMA integrity_check").fetchone()[0]
    if result != "ok":
        raise RuntimeError(f"SQLite integrity check failed: {result}")
    return max(version for version, _ in migrations)


@contextmanager
def database(path: Path) -> Iterator[sqlite3.Connection]:
    conn = connect(path)
    try:
        migrate(conn)
        yield conn
    finally:
        conn.close()
