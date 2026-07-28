from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from mouselearn.models import (
    ConditionalFlowConfig, PromotionError, build_baseline_model, load_generator, promote_model,
    train_experimental_conditional_flow,
)
from mouselearn.storage.bootstrap import initialize
from mouselearn.storage.database import connect
from mouselearn.storage.repositories import Repositories


def _completed_processed_run(root: Path, database: Path) -> str:
    snapshot_id, run_id = "snapshot-baseline", "run-baseline"
    trial_ids: list[str] = []
    records = []
    conn = connect(database)
    try:
        for session_index, split in enumerate(("train", "validation", "test")):
            session_id = f"session-{session_index}"
            environment = {"virtual_screen_physical_bounds": {"left": 0, "top": 0, "width": 1920, "height": 1080}, "qt_device_pixel_ratio": 1.0}
            conn.execute("INSERT INTO recording_sessions(id,status,created_at,finished_at) VALUES(?,?,?,?)", (session_id, "completed", "now", "now"))
            conn.execute(
                """INSERT INTO collection_session_details(session_id,display_name,mode,state,planned_trials,random_seed,
                          config_json,environment_json,started_at,ended_at,updated_at) VALUES(?,?,?,?,?,?,?,?,?,?,?)""",
                (session_id, session_id, "standard", "completed", 6, session_index, "{}", json.dumps(environment), "now", "now", "now"),
            )
            for local_index in range(6):
                ordinal = len(trial_ids)
                trial_id = f"trial-{ordinal}"
                trial_ids.append(trial_id)
                angle = ordinal * math.pi / 9
                start_x, start_y = 500 + local_index, 400 + session_index
                distance, radius = 200 + local_index * 10, 20 + ordinal % 3
                target_x = round(start_x + math.cos(angle) * distance)
                target_y = round(start_y + math.sin(angle) * distance)
                condition = {
                    "distance_px": distance, "radius_px": radius, "angle_degrees": math.degrees(angle) % 360,
                    "screen_region": "center", "difficulty_band": "medium", "target_x": target_x, "target_y": target_y,
                    "monitor_id": "test", "collection_protocol_version": 3,
                    "target_sampling_strategy": "continuous_uniform_feasible_v3", "reaction_time_confidence": "legacy_render_unconfirmed",
                }
                conn.execute("INSERT INTO trials(id,session_id,status,created_at) VALUES(?,?,?,?)", (trial_id, session_id, "completed", "now"))
                conn.execute(
                    """INSERT INTO trial_details(trial_id,condition_json,state,target_appeared_ns,start_screen_x,start_screen_y,
                              valid_click_ns,ended_ns,end_reason,clicks_json) VALUES(?,?,?,?,?,?,?,?,?,?)""",
                    (trial_id, json.dumps(condition), "completed", ordinal * 1000, start_x, start_y,
                     ordinal * 1000 + 100, ordinal * 1000 + 100, "valid_click", "[]"),
                )
                positions = [[index / 63, 0.025 * math.sin(math.pi * index / 63) * ((ordinal % 3) - 1)] for index in range(64)]
                positions[-1] = [1.0, 0.0]
                records.append({
                    "trial_id": trial_id, "session_id": session_id, "split": split, "ordinal": ordinal,
                    "canonical_positions": positions, "log_total_movement_duration": math.log(200_000_000 + ordinal * 2_000_000),
                })
        conn.execute("INSERT INTO dataset_snapshots(id,status,created_at) VALUES(?,?,?)", (snapshot_id, "draft", "now"))
        conn.execute(
            """INSERT INTO dataset_snapshot_details(snapshot_id,name,description,ordered_trial_ids_json,raw_session_hashes_json,
                      preprocessing_config_json,split_config_json,warnings_json,feature_schema_version,code_revision,
                      manifest_sha256,manifest_relative_path,trial_count,session_count) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (snapshot_id, "baseline", "", json.dumps(trial_ids), "[]", "{}", "{}", "[]", 2, "test", "0" * 64, "datasets/manifest.json", len(trial_ids), 3),
        )
        for ordinal, trial_id in enumerate(trial_ids):
            split = "train" if ordinal < 6 else "validation" if ordinal < 12 else "test"
            conn.execute("INSERT INTO dataset_snapshot_trials(snapshot_id,trial_id,session_id,split,ordinal) VALUES(?,?,?,?,?)", (snapshot_id, trial_id, f"session-{ordinal // 6}", split, ordinal))
        conn.execute("UPDATE dataset_snapshots SET status='ready' WHERE id=?", (snapshot_id,))
        processed = root / "datasets" / snapshot_id / "preprocessing" / run_id / "processed.parquet"
        processed.parent.mkdir(parents=True)
        pq.write_table(pa.Table.from_pylist(records), processed)
        digest = hashlib.sha256(processed.read_bytes()).hexdigest()
        conn.execute("INSERT INTO preprocessing_runs(id,status,created_at) VALUES(?,?,?)", (run_id, "completed", "now"))
        conn.execute(
            """INSERT INTO preprocessing_run_details(run_id,snapshot_id,config_json,code_revision,processed_relative_path,
                      processed_sha256,processed_trial_count,skipped_trial_count,started_at,finished_at) VALUES(?,?,?,?,?,?,?,?,?,?)""",
            (run_id, snapshot_id, "{}", "test", processed.relative_to(root).as_posix(), digest, len(records), 0, "now", "now"),
        )
    finally:
        conn.close()
    return run_id


def test_baseline_build_publishes_loadable_snapshot_bound_artifact(data_root: Path) -> None:
    root, database, _ = initialize(data_root)
    run_id = _completed_processed_run(root, database)
    result = build_baseline_model(root, database, run_id, "pca_mixture", {"latent_dimension": 5, "mixture_component_count": 3})
    assert result["validation"]["passed"]
    conn = connect(database)
    try:
        model = Repositories(conn).baseline_model(result["model_id"])
    finally:
        conn.close()
    assert model["status"] == "ready"
    source = root / model["manifest_relative_path"]
    generator = load_generator(source.parent)
    assert generator.model_type == "pca_mixture"
    manifest = json.loads(source.read_text(encoding="utf-8"))
    assert manifest["dataset_snapshot_id"] == "snapshot-baseline"
    assert manifest["preprocessing_run_id"] == run_id
    assert manifest["validation_sha256"] == model["validation_sha256"]
    promote_model(root, database, model["id"])
    conn = connect(database)
    try:
        assert Repositories(conn).registry_models()[0]["lifecycle"] == "active"
    finally:
        conn.close()
    validation = root / model["validation_relative_path"]
    validation.write_text("{}\n", encoding="utf-8")
    with pytest.raises(PromotionError, match="hash changed"):
        promote_model(root, database, model["id"])


def test_experimental_flow_is_ready_but_remains_unvalidated_candidate(data_root: Path) -> None:
    root, database, _ = initialize(data_root)
    run_id = _completed_processed_run(root, database)
    config = ConditionalFlowConfig(
        hidden_size=16, hidden_layers=1, epochs=2, batch_size=18, checkpoint_every=1,
        solver_steps=2, training_scope="all", validation_mode="none", condition_mode="zero",
    )
    result = train_experimental_conditional_flow(root, database, run_id, config)
    conn = connect(database)
    try:
        model = Repositories(conn).baseline_model(result["model_id"])
        registry_model = next(item for item in Repositories(conn).registry_models() if item["id"] == model["id"])
    finally:
        conn.close()
    assert model["status"] == "ready"
    assert model["validation_relative_path"] is None
    assert registry_model["lifecycle"] == "candidate"
    assert result["training_report"]["training_sample_count"] == 18
    manifest = json.loads((root / model["manifest_relative_path"]).read_text(encoding="utf-8"))
    assert manifest["validation_status"] == "skipped_by_request"
    with pytest.raises(PromotionError, match="no completed validation"):
        promote_model(root, database, model["id"])


def test_experiment_cancellation_is_tied_to_worker_job(data_root: Path) -> None:
    root, database, _ = initialize(data_root)
    run_id = _completed_processed_run(root, database)
    conn = connect(database)
    try:
        repos = Repositories(conn)
        job_id = repos.create_job("conditional_flow")
        experiment_id = repos.create_experiment("cancel me", "snapshot-baseline", run_id, {"epochs": 10}, 42, job_id)
        repos.start_experiment(experiment_id)
        repos.cancel_experiments_for_job(job_id)
        experiment = next(item for item in repos.experiments() if item["id"] == experiment_id)
    finally:
        conn.close()
    assert experiment["status"] == "cancelled"
    assert experiment["error"] == "Cancelled by user"
