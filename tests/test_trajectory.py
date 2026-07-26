from __future__ import annotations

from mouselearn.collection.native import NativeMouseEvent
from mouselearn.collection.parquet import BoundedParquetWriter
from mouselearn.collection.trajectory import load_trial_trajectory
from mouselearn.domain.collection import ClickRecord, CollectionSessionPlan, TargetCondition, TrialFinalization, TrialPlan
from mouselearn.storage.bootstrap import initialize
from mouselearn.storage.database import connect
from mouselearn.storage.repositories import Repositories


def test_trajectory_loader_uses_only_target_visible_interval(data_root) -> None:
    root, db, _ = initialize(data_root)
    conn = connect(db)
    try:
        repo = Repositories(conn)
        session_id = repo.create_collection_session(CollectionSessionPlan(display_name="trajectory", planned_trials=1, random_seed=1))
        repo.transition_collection_session(session_id, "active")
        trial_id = repo.create_trial(TrialPlan(
            session_id=session_id,
            condition=TargetCondition(distance_px=100, radius_px=20, angle_degrees=0, screen_region="center", difficulty_band="medium", target_x=110, target_y=100, monitor_id="one"),
            target_appeared_ns=100, start_screen_x=10, start_screen_y=10,
        ))
        repo.finalize_trial(trial_id, TrialFinalization(
            state="completed", end_reason="valid_click", ended_ns=200,
            clicks=(ClickRecord(timestamp_ns=200, screen_x=110, screen_y=100, is_valid=True),),
        ))
        writer = BoundedParquetWriter(root / "raw_sessions", session_id, qpc_frequency_hz=1_000_000_000)
        writer.start()
        assert writer.submit([
            NativeMouseEvent(90, 0, 0, 1, 1, 0, 0, 1, 1, 0),
            NativeMouseEvent(100, 0, 0, 10, 10, 0, 0, 1, 1, 0),
            NativeMouseEvent(150, 0, 0, 60, 50, 0, 0, 1, 1, 0),
            NativeMouseEvent(200, 0, 0, 110, 100, 1, 0, 1, 1, 0),
            NativeMouseEvent(210, 0, 0, 120, 110, 0, 0, 1, 1, 0),
        ])
        repo.record_raw_event_file(session_id, writer.finalize())
        trajectory = load_trial_trajectory(root / "raw_sessions", repo.trial(trial_id), repo.raw_event_files(session_id))
        assert trajectory["raw_point_count"] == 3
        assert [(point["x"], point["y"]) for point in trajectory["points"]] == [(10, 10), (60, 50), (110, 100)]
        assert trajectory["start"] == {"x": 10, "y": 10}
        assert trajectory["target"]["x"] == 110
    finally:
        conn.close()
