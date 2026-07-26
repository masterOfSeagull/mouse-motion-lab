"""Load an immutable preprocessing artifact with its snapshot-bound conditions."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from mouselearn.storage.database import connect, migrate

from .base import GenerationRequest, ProcessedDataset, condition_vector


class ProcessedDatasetError(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1_048_576), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _request(condition: dict[str, Any], start_x: float, start_y: float, environment: dict[str, Any]) -> GenerationRequest:
    bounds = environment.get("virtual_screen_physical_bounds")
    if not isinstance(bounds, dict) or not all(name in bounds for name in ("left", "top", "width", "height")):
        raise ProcessedDatasetError("recording session lacks persisted virtual-screen physical bounds")
    return GenerationRequest(
        start_x=start_x, start_y=start_y, target_center_x=float(condition["target_x"]),
        target_center_y=float(condition["target_y"]), target_radius=float(condition["radius_px"]),
        virtual_desktop_left=float(bounds["left"]), virtual_desktop_top=float(bounds["top"]),
        virtual_desktop_width=float(bounds["width"]), virtual_desktop_height=float(bounds["height"]),
        dpi_scale=float(environment.get("qt_device_pixel_ratio", 1.0)),
    )


def load_processed_dataset(root: Path, database: Path, preprocessing_run_id: str) -> ProcessedDataset:
    """Verify the processed artifact hash and join only immutable snapshot membership."""
    conn = connect(database)
    try:
        migrate(conn)
        run = conn.execute(
            """SELECT r.status,d.snapshot_id,d.processed_relative_path,d.processed_sha256
                 FROM preprocessing_runs r JOIN preprocessing_run_details d ON d.run_id=r.id
                 WHERE r.id=?""", (preprocessing_run_id,),
        ).fetchone()
        if run is None:
            raise ProcessedDatasetError(f"unknown preprocessing run: {preprocessing_run_id}")
        if run["status"] != "completed" or not run["processed_relative_path"] or not run["processed_sha256"]:
            raise ProcessedDatasetError("model fitting requires a completed preprocessing run")
        root = root.resolve()
        processed_path = (root / run["processed_relative_path"]).resolve()
        if not processed_path.is_relative_to(root) or not processed_path.is_file():
            raise ProcessedDatasetError("processed artifact is missing or outside the data root")
        if _sha256(processed_path) != run["processed_sha256"]:
            raise ProcessedDatasetError("processed artifact hash changed")
        rows = conn.execute(
            """SELECT st.trial_id,st.split,st.ordinal,td.condition_json,td.start_screen_x,td.start_screen_y,
                      cs.environment_json
                 FROM dataset_snapshot_trials st
                 JOIN trial_details td ON td.trial_id=st.trial_id
                 JOIN collection_session_details cs ON cs.session_id=st.session_id
                 WHERE st.snapshot_id=? ORDER BY st.ordinal""", (run["snapshot_id"],),
        ).fetchall()
        metadata = {row["trial_id"]: row for row in rows}
    finally:
        conn.close()
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise ProcessedDatasetError("PyArrow is required to load processed datasets") from exc
    table = pq.read_table(processed_path)
    records = table.to_pylist()
    conditions, outputs, requests, splits, trial_ids = [], [], [], [], []
    for record in records:
        trial_id = str(record["trial_id"])
        row = metadata.get(trial_id)
        if row is None:
            raise ProcessedDatasetError(f"processed trial is not in its immutable snapshot: {trial_id}")
        if row["start_screen_x"] is None or row["start_screen_y"] is None:
            raise ProcessedDatasetError(f"processed trial lacks a recorded start: {trial_id}")
        try:
            request = _request(
                json.loads(row["condition_json"]), float(row["start_screen_x"]), float(row["start_screen_y"]),
                json.loads(row["environment_json"]),
            )
            canonical = np.asarray(record["canonical_positions"], dtype=np.float64)
            if canonical.shape != (64, 2):
                raise ProcessedDatasetError(f"processed trial has the wrong position shape: {trial_id}")
            output = np.concatenate((canonical.reshape(-1), [float(record["log_total_movement_duration"])]))
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ProcessedDatasetError(f"invalid processed metadata for trial {trial_id}: {exc}") from exc
        conditions.append(condition_vector(request))
        outputs.append(output)
        requests.append(request)
        splits.append(str(row["split"]))
        trial_ids.append(trial_id)
    if len(records) != len(metadata):
        missing = sorted(set(metadata) - set(trial_ids))
        raise ProcessedDatasetError(f"processed artifact omits {len(missing)} immutable snapshot trial(s)")
    return ProcessedDataset(
        np.asarray(conditions), np.asarray(outputs), tuple(requests), tuple(splits), tuple(trial_ids),
        str(run["snapshot_id"]), preprocessing_run_id,
    )


__all__ = ["ProcessedDatasetError", "load_processed_dataset"]
