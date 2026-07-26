from __future__ import annotations

import hashlib
import json
import sqlite3

import pytest

from mouselearn.datasets.snapshots import DatasetBuildError, build_dataset_snapshot
from mouselearn.domain.collection import ClickRecord, CollectionSessionPlan, RawEventFileReference, TargetCondition, TrialFinalization, TrialPlan
from mouselearn.domain.dataset import DatasetSnapshotPlan, SessionHeldOutSplit, session_held_out_assignments
from mouselearn.preprocessing import preprocess_dataset_snapshot
from mouselearn.storage.bootstrap import initialize
from mouselearn.storage.database import connect
from mouselearn.storage.repositories import Repositories


def _completed_session(root, repos: Repositories, index: int) -> str:
    session_id = repos.create_collection_session(CollectionSessionPlan(display_name=f"session {index}", planned_trials=1, random_seed=index))
    repos.transition_collection_session(session_id, "active")
    target = TargetCondition(
        distance_px=200, radius_px=30, angle_degrees=0, screen_region="center", difficulty_band="medium",
        target_x=500, target_y=300, monitor_id="test-monitor", collection_protocol_version=3,
        target_sampling_strategy="continuous_uniform_feasible_v3",
    )
    trial_id = repos.create_trial(TrialPlan(
        session_id=session_id, condition=target, target_appeared_ns=1_000 + index * 100,
        start_screen_x=300, start_screen_y=300,
    ))
    repos.finalize_trial(trial_id, TrialFinalization(
        state="completed", end_reason="valid_click", ended_ns=1_100 + index * 100,
        clicks=(ClickRecord(timestamp_ns=1_100 + index * 100, screen_x=500, screen_y=300, is_valid=True),),
    ))
    raw_path = root / "raw_sessions" / session_id / "events.parquet"
    raw_path.parent.mkdir(parents=True)
    payload = f"raw evidence {index}".encode()
    raw_path.write_bytes(payload)
    repos.record_raw_event_file(session_id, RawEventFileReference(
        relative_path=f"{session_id}/events.parquet", event_count=1, first_timestamp_ns=1_000,
        last_timestamp_ns=1_100, qpc_frequency_hz=1_000_000, byte_count=len(payload), sha256=hashlib.sha256(payload).hexdigest(),
    ))
    repos.transition_collection_session(session_id, "completed")
    return session_id


def test_snapshot_is_deterministic_immutable_and_session_held_out(data_root) -> None:
    root, db, _ = initialize(data_root)
    conn = connect(db)
    try:
        repos = Repositories(conn)
        sessions = [_completed_session(root, repos, index) for index in range(3)]
    finally:
        conn.close()

    plan = DatasetSnapshotPlan(name="baseline", session_ids=tuple(sessions), preprocessing_config={"dedupe": "last"})
    first = build_dataset_snapshot(root, db, plan)
    second = build_dataset_snapshot(root, db, plan)
    assert first["status"] == "ready"
    assert first["manifest_sha256"] == second["manifest_sha256"]
    manifest = json.loads((root / first["manifest_relative_path"]).read_text(encoding="utf-8"))
    assert manifest["ordered_trial_ids"] == first["ordered_trial_ids"]
    assert len(first["splits"]) == 3
    assert {row["session_id"]: row["split"] for row in first["splits"]}
    assert set(row["split"] for row in first["splits"]).issubset({"train", "validation", "test"})
    conn = connect(db)
    try:
        with pytest.raises(sqlite3.IntegrityError, match="immutable"):
            conn.execute("UPDATE dataset_snapshot_details SET name='changed' WHERE snapshot_id=?", (first["id"],))
    finally:
        conn.close()


def test_session_held_out_warns_when_independence_is_impossible() -> None:
    assignments, warnings = session_held_out_assignments(["session-a"], SessionHeldOutSplit(seed=3))
    assert assignments == {"session-a": "train"}
    assert warnings and "not independent" in warnings[0]


def test_session_held_out_uses_the_configured_fractions() -> None:
    session_ids = [f"session-{index}" for index in range(10)]
    split = SessionHeldOutSplit(seed=9, train_fraction=0.5, validation_fraction=0.3, test_fraction=0.2)
    assignments, warnings = session_held_out_assignments(session_ids, split)
    assert not warnings
    assert list(assignments.values()).count("train") == 5
    assert list(assignments.values()).count("validation") == 3
    assert list(assignments.values()).count("test") == 2


def test_snapshot_failure_cleans_up_its_draft_and_directory(data_root) -> None:
    root, db, _ = initialize(data_root)
    conn = connect(db)
    try:
        session_id = _completed_session(root, Repositories(conn), 1)
    finally:
        conn.close()
    raw_path = root / "raw_sessions" / session_id / "events.parquet"
    raw_path.write_bytes(b"changed after recording")
    with pytest.raises(DatasetBuildError, match="hash changed"):
        build_dataset_snapshot(root, db, DatasetSnapshotPlan(name="should fail"))
    conn = connect(db)
    try:
        assert conn.execute("SELECT count(*) FROM dataset_snapshots").fetchone()[0] == 0
    finally:
        conn.close()
    assert list((root / "datasets").iterdir()) == []


def test_preprocessing_writes_deterministic_representation_and_report(data_root) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    root, db, _ = initialize(data_root)
    conn = connect(db)
    try:
        repos = Repositories(conn)
        sessions = [_completed_session(root, repos, index) for index in range(3)]
        for index, session_id in enumerate(sessions):
            raw_path = root / "raw_sessions" / session_id / "events.parquet"
            start_timestamp = 1_000 + index * 100
            table = pa.table({
                "timestamp_ns": [start_timestamp, start_timestamp + 50, start_timestamp + 100], "raw_dx": [0, 100, 100], "raw_dy": [0, 0, 0],
                "screen_x": [300, 400, 500], "screen_y": [300, 300, 300], "button_flags": [0, 0, 1],
                "event_flags": [0, 0, 0], "device_handle": [1, 1, 1], "foreground_collection_window": [True, True, True],
            })
            pq.write_table(table, raw_path, compression="zstd")
            payload = raw_path.read_bytes()
            conn.execute(
                "UPDATE raw_event_files SET event_count=?, byte_count=?, sha256=? WHERE session_id=?",
                (3, len(payload), hashlib.sha256(payload).hexdigest(), session_id),
            )
    finally:
        conn.close()
    snapshot = build_dataset_snapshot(root, db, DatasetSnapshotPlan(name="preprocess", session_ids=tuple(sessions)))
    first = preprocess_dataset_snapshot(root, db, snapshot["id"])
    second = preprocess_dataset_snapshot(root, db, snapshot["id"])
    assert first["processed_trial_count"] == 3
    assert first["skipped_trial_count"] == 0
    assert first["resampling"]["max_error"] < 1e-6
    assert json.loads((root / first["report_path"]).read_text())["processed_trial_count"] == 3
    records = pq.read_table(root / first["processed_path"]).to_pylist()
    assert len(records[0]["canonical_positions"]) == 64
    assert records[0]["canonical_positions"][0] == [0.0, 0.0]
    assert records == pq.read_table(root / second["processed_path"]).to_pylist()
