from __future__ import annotations

import sqlite3

import pytest

from mouselearn.domain.collection import (
    CaptureHealthRecord,
    ClickRecord,
    CollectionPhaseMarker,
    CollectionSessionPlan,
    RawEventFileReference,
    TargetCondition,
    TrialFinalization,
    TrialPlan,
)
from mouselearn.storage.bootstrap import initialize
from mouselearn.storage.database import MIGRATIONS, connect, migrate
from mouselearn.storage.repositories import Repositories


def test_first_run_and_migrations_are_idempotent(data_root) -> None:
    root, db, version = initialize(data_root)
    assert version == 7
    assert db == root / "app.db"
    assert all((root / name).is_dir() for name in (
        "logs", "raw_sessions", "datasets", "experiments", "models", "exports", "cache", "temp",
    ))
    conn = connect(db)
    try:
        assert migrate(conn) == 7
        assert conn.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
    finally:
        conn.close()


def test_migration_four_upgrades_a_version_one_database(data_root) -> None:
    db = data_root / "app.db"
    conn = connect(db)
    try:
        conn.executescript(MIGRATIONS[0][1])
        conn.execute("CREATE TABLE schema_migrations (version INTEGER PRIMARY KEY, applied_at TEXT NOT NULL)")
        conn.execute("INSERT INTO schema_migrations(version, applied_at) VALUES(1, '2026-07-26T00:00:00+00:00')")
        assert migrate(conn) == 7
        assert conn.execute("SELECT count(*) FROM pragma_table_info('raw_event_files')").fetchone()[0] > 0
        assert conn.execute("SELECT count(*) FROM pragma_table_info('trial_reviews')").fetchone()[0] > 0
        assert conn.execute("SELECT count(*) FROM pragma_table_info('collection_phase_markers')").fetchone()[0] > 0
        assert conn.execute("SELECT count(*) FROM pragma_table_info('dataset_snapshot_details')").fetchone()[0] > 0
        assert conn.execute("SELECT count(*) FROM pragma_table_info('preprocessing_run_details')").fetchone()[0] > 0
    finally:
        conn.close()


def test_failed_migration_rolls_back_all_schema_changes(data_root) -> None:
    conn = connect(data_root / "atomic.db")
    try:
        failing = ((1, "CREATE TABLE must_not_remain(id INTEGER); CREATE TABLE must_not_remain(id INTEGER);"),)
        with pytest.raises(sqlite3.OperationalError):
            migrate(conn, migrations=failing)
        assert conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='must_not_remain'").fetchone() is None
        assert conn.execute("SELECT count(*) FROM schema_migrations").fetchone()[0] == 0
    finally:
        conn.close()


def test_collection_repositories_persist_a_typed_session(data_root) -> None:
    _, db, _ = initialize(data_root)
    conn = connect(db)
    try:
        repo = Repositories(conn)
        session_id = repo.create_collection_session(CollectionSessionPlan(
            display_name="Baseline coverage", mode="balanced_coverage", planned_trials=3, random_seed=7,
            config={"inter_trial_delay_ms": [400, 1200]}, environment={"dpi": 96},
        ))
        assert repo.collection_session(session_id)["state"] == "planned"
        repo.transition_collection_session(session_id, "active")
        condition = TargetCondition(
            distance_px=180, radius_px=24, angle_degrees=90, screen_region="center", difficulty_band="medium",
            target_x=960, target_y=540, monitor_id="monitor-1",
        )
        trial_id = repo.create_trial(TrialPlan(session_id=session_id, condition=condition, target_appeared_ns=1_000, start_screen_x=800, start_screen_y=500))
        repo.finalize_trial(trial_id, TrialFinalization(
            state="completed", end_reason="valid_click", ended_ns=1_900,
            clicks=(
                ClickRecord(timestamp_ns=1_400, screen_x=900, screen_y=500, is_valid=False),
                ClickRecord(timestamp_ns=1_800, screen_x=960, screen_y=540, is_valid=True),
            ),
        ))
        trial = repo.trial(trial_id)
        assert trial["condition"]["target_x"] == 960
        assert (trial["start_screen_x"], trial["start_screen_y"]) == (800, 500)
        assert trial["valid_click_ns"] == 1_800

        file_id = repo.record_raw_event_file(session_id, RawEventFileReference(
            relative_path=f"{session_id}/events-0001.parquet", event_count=22, first_timestamp_ns=1_000,
            last_timestamp_ns=1_900, qpc_frequency_hz=10_000_000, byte_count=4096,
        ))
        health_id = repo.record_capture_health(session_id, CaptureHealthRecord(
            severity="warning", code="buffer_high_watermark", occurred_at_ns=1_500, detail={"ratio": 0.8},
        ))
        assert repo.raw_event_files(session_id)[0]["id"] == file_id
        assert conn.execute("SELECT session_id FROM capture_health_events WHERE id=?", (health_id,)).fetchone()[0] == session_id
        repo.transition_collection_session(session_id, "completed")
        assert repo.collection_session(session_id)["legacy_status"] == "completed"
    finally:
        conn.close()


def test_collection_contracts_reject_bad_paths_and_lifecycle(data_root) -> None:
    with pytest.raises(ValueError, match="relative"):
        RawEventFileReference(relative_path="../events.parquet", event_count=0, qpc_frequency_hz=1, byte_count=0)
    with pytest.raises(ValueError, match="valid click"):
        TrialFinalization(state="completed", end_reason="valid_click", ended_ns=1)

    _, db, _ = initialize(data_root)
    conn = connect(db)
    try:
        repo = Repositories(conn)
        session_id = repo.create_collection_session(CollectionSessionPlan(display_name="one", planned_trials=1, random_seed=0))
        with pytest.raises(ValueError, match="invalid collection session transition"):
            repo.transition_collection_session(session_id, "completed")
    finally:
        conn.close()


def test_review_discard_is_reversible_and_preserves_trials(data_root) -> None:
    _, db, _ = initialize(data_root)
    conn = connect(db)
    try:
        repo = Repositories(conn)
        session_id = repo.create_collection_session(CollectionSessionPlan(display_name="review me", planned_trials=1, random_seed=0))
        repo.transition_collection_session(session_id, "active")
        trial_id = repo.create_trial(TrialPlan(
            session_id=session_id,
            condition=TargetCondition(distance_px=80, radius_px=24, angle_degrees=0, screen_region="center", difficulty_band="low", target_x=100, target_y=100, monitor_id="one"),
            target_appeared_ns=10, start_screen_x=0, start_screen_y=0,
        ))
        repo.finalize_trial(trial_id, TrialFinalization(
            state="completed", end_reason="valid_click", ended_ns=20,
            clicks=(ClickRecord(timestamp_ns=20, screen_x=100, screen_y=100, is_valid=True),),
        ))
        repo.transition_collection_session(session_id, "completed")

        repo.record_phase_marker(session_id, CollectionPhaseMarker(phase="target_visible", timestamp_ns=10, screen_x=0, screen_y=0), trial_id)
        repo.record_phase_marker(session_id, CollectionPhaseMarker(phase="trial_completed", timestamp_ns=20, screen_x=100, screen_y=100), trial_id)
        assert [row["phase"] for row in repo.phase_markers_for_trial(trial_id)] == ["target_visible", "trial_completed"]

        assert repo.collection_sessions_for_review()[0]["review_disposition"] == "retained"
        repo.set_trial_review(trial_id, "discarded", "misclick pattern")
        assert repo.trials_for_review(session_id)[0]["review_disposition"] == "discarded"
        repo.set_trial_review(trial_id, "retained")
        assert repo.trials_for_review(session_id)[0]["review_disposition"] == "retained"
        repo.set_session_review(session_id, "discarded", "test session")
        assert repo.collection_sessions_for_review()[0]["review_disposition"] == "discarded"
        assert repo.trial(trial_id)["status"] == "completed"  # logical discard never deletes evidence
        repo.set_session_review(session_id, "retained")
        assert repo.collection_sessions_for_review()[0]["review_disposition"] == "retained"
    finally:
        conn.close()


def test_legacy_sessions_retain_data_but_recompute_realized_geometry(data_root) -> None:
    _, db, _ = initialize(data_root)
    conn = connect(db)
    try:
        repo = Repositories(conn)
        session_id = repo.create_collection_session(CollectionSessionPlan(display_name="legacy", planned_trials=1, random_seed=0))
        repo.transition_collection_session(session_id, "active")
        trial_id = repo.create_trial(TrialPlan(
            session_id=session_id,
            condition=TargetCondition(distance_px=1, radius_px=20, angle_degrees=0, screen_region="center", difficulty_band="low", target_x=300, target_y=200, monitor_id="one"),
            target_appeared_ns=10, start_screen_x=100, start_screen_y=200,
        ))
        repo.finalize_trial(trial_id, TrialFinalization(
            state="completed", end_reason="valid_click", ended_ns=20,
            clicks=(ClickRecord(timestamp_ns=20, screen_x=300, screen_y=200, is_valid=True),),
        ))
        repo.transition_collection_session(session_id, "completed")
        repo.set_collection_quality(session_id, "legacy", ["test"])
        assert repo.reconcile_legacy_collection_data() == 1
        trial = repo.trial(trial_id)
        assert trial["condition"]["distance_px"] == 200
        assert trial["condition"]["angle_degrees"] == 0
        assert trial["condition"]["reaction_time_confidence"] == "legacy_render_unconfirmed"
        repo.record_phase_marker(session_id, CollectionPhaseMarker(phase="session_cancelled", timestamp_ns=21))
        assert repo.phase_markers_for_trial(trial_id) == []
    finally:
        conn.close()


def test_repositories_and_default_model_constraint(data_root) -> None:
    _, db, _ = initialize(data_root)
    conn = connect(db)
    try:
        repo = Repositories(conn)
        repo.set_setting("theme", {"dark": True})
        assert repo.get_setting("theme") == {"dark": True}
        job_id = repo.create_job("diagnostic")
        repo.update_job(job_id, status="running", progress=12, stage="database")
        assert repo.job(job_id)["progress"] == 12
        now = "2026-01-01T00:00:00+00:00"
        conn.execute("INSERT INTO models(id,name,status,is_default,created_at) VALUES('a','a','ready',1,?)", (now,))
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute("INSERT INTO models(id,name,status,is_default,created_at) VALUES('b','b','ready',1,?)", (now,))
    finally:
        conn.close()
