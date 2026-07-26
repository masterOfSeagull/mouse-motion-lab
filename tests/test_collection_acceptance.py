from __future__ import annotations

from collections import Counter

from mouselearn.collection.native import NativeMouseEvent
from mouselearn.collection.parquet import BoundedParquetWriter
from mouselearn.collection.targets import BalancedTargetScheduler
from mouselearn.domain.collection import ClickRecord, CollectionSessionPlan, TargetCondition, TrialFinalization, TrialPlan
from mouselearn.storage.bootstrap import initialize
from mouselearn.storage.database import connect
from mouselearn.storage.repositories import Repositories


def test_balanced_scheduler_is_reproducible_and_keeps_targets_on_canvas() -> None:
    first = [BalancedTargetScheduler(7).next(1280, 720)]
    scheduler = BalancedTargetScheduler(7)
    targets = [scheduler.next(1280, 720) for _ in range(500)]
    assert targets[0] == first[0]
    assert all(target.radius <= target.x <= 1280 - target.radius for target in targets)
    assert all(target.radius <= target.y <= 720 - target.radius for target in targets)
    radius_counts = Counter(target.radius for target in targets)
    region_counts = Counter(target.screen_region for target in targets)
    assert max(radius_counts.values()) - min(radius_counts.values()) <= 1
    assert max(region_counts.values()) - min(region_counts.values()) <= 1


def test_synthetic_500_trial_session_persists_without_loss(data_root) -> None:
    root, db, _ = initialize(data_root)
    conn = connect(db)
    try:
        repo = Repositories(conn)
        session_id = repo.create_collection_session(CollectionSessionPlan(
            display_name="500-trial acceptance", planned_trials=500, random_seed=19,
        ))
        repo.transition_collection_session(session_id, "active")
        scheduler = BalancedTargetScheduler(19)
        events: list[NativeMouseEvent] = []
        for index in range(500):
            target = scheduler.next(1280, 720)
            appeared = 1_000_000 + index * 10_000
            trial_id = repo.create_trial(TrialPlan(
                session_id=session_id,
                condition=TargetCondition(
                    distance_px=target.distance_px, radius_px=target.radius, angle_degrees=target.angle_degrees,
                    screen_region=target.screen_region, difficulty_band=target.difficulty_band,
                    target_x=target.x, target_y=target.y, monitor_id="synthetic",
                ),
                target_appeared_ns=appeared, start_screen_x=640, start_screen_y=360,
            ))
            click = ClickRecord(timestamp_ns=appeared + 1_000, screen_x=target.x, screen_y=target.y, is_valid=True)
            repo.finalize_trial(trial_id, TrialFinalization(
                state="completed", end_reason="valid_click", ended_ns=click.timestamp_ns, clicks=(click,),
            ))
            events.append(NativeMouseEvent(index + 1, 1, 1, target.x, target.y, 1, 0, 1, 1, 0))

        writer = BoundedParquetWriter(root / "raw_sessions", session_id, qpc_frequency_hz=1_000_000_000)
        writer.start()
        assert writer.submit(events)
        repo.record_raw_event_file(session_id, writer.finalize())
        repo.transition_collection_session(session_id, "completed")

        assert conn.execute("SELECT count(*) FROM trials WHERE session_id=? AND status='completed'", (session_id,)).fetchone()[0] == 500
        assert repo.raw_event_files(session_id)[0]["event_count"] == 500
        assert repo.collection_session(session_id)["state"] == "completed"
    finally:
        conn.close()
