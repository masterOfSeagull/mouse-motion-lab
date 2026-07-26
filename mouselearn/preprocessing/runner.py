"""Snapshot-bound preprocessing.  Raw files are verified, never modified."""
from __future__ import annotations

import hashlib
import json
import math
import shutil
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from mouselearn.datasets.snapshots import _file_digest, current_code_revision
from mouselearn.representation.canonical import CanonicalTransform
from mouselearn.storage.database import connect, migrate
from mouselearn.storage.repositories import Repositories


class PreprocessingError(RuntimeError):
    pass


@dataclass(frozen=True)
class PreprocessingSpec:
    schema_version: int = 2
    equal_time_position_count: int = 64
    onset_displacement_px: float = 3.0

    def __post_init__(self) -> None:
        if self.schema_version != 2:
            raise ValueError("unsupported preprocessing schema version")
        if self.equal_time_position_count < 2:
            raise ValueError("equal_time_position_count must be at least 2")
        if self.onset_displacement_px < 0:
            raise ValueError("onset_displacement_px must be nonnegative")

    def config(self) -> dict[str, Any]:
        return asdict(self)


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _digest_file(path: Path) -> str:
    return _file_digest(path)


def _valid_click(clicks_json: str) -> tuple[int, int, int]:
    clicks = json.loads(clicks_json)
    valid = [click for click in clicks if click.get("is_valid")]
    if not valid:
        raise PreprocessingError("completed trial has no valid click")
    click = valid[-1]
    return int(click["timestamp_ns"]), int(click["screen_x"]), int(click["screen_y"])


def _strict_points(
    start_timestamp_ns: int, start: tuple[int, int], click: tuple[int, int, int], raw_events: list[tuple[int, int, int]],
) -> tuple[list[int], list[tuple[float, float]]]:
    """Keep phase-bounded motion, replacing same-timestamp samples deterministically."""
    click_timestamp, click_x, click_y = click
    by_timestamp: dict[int, tuple[float, float]] = {start_timestamp_ns: (float(start[0]), float(start[1]))}
    for timestamp, x, y in raw_events:
        if start_timestamp_ns < timestamp < click_timestamp:
            by_timestamp[timestamp] = (float(x), float(y))
    by_timestamp[click_timestamp] = (float(click_x), float(click_y))
    timestamps = sorted(by_timestamp)
    if len(timestamps) < 2:
        raise PreprocessingError("trial has no positive-duration movement interval")
    return timestamps, [by_timestamp[timestamp] for timestamp in timestamps]


def _movement_onset_index(points: list[tuple[float, float]], threshold_px: float) -> int:
    """Ignore sub-threshold sensor drift without making reaction time a model target."""
    origin = points[0]
    for index, point in enumerate(points[1:], start=1):
        if math.dist(origin, point) >= threshold_px:
            return index
    return 0


def _interpolate_position(
    timestamps: list[int], points: list[tuple[float, float]], timestamp: int,
) -> tuple[float, float]:
    if timestamp <= timestamps[0]:
        return points[0]
    if timestamp >= timestamps[-1]:
        return points[-1]
    for index in range(1, len(timestamps)):
        if timestamps[index] >= timestamp:
            left_time, right_time = timestamps[index - 1], timestamps[index]
            left, right = points[index - 1], points[index]
            span = right_time - left_time
            if span <= 0:
                return right
            weight = (timestamp - left_time) / span
            return (left[0] + (right[0] - left[0]) * weight, left[1] + (right[1] - left[1]) * weight)
    return points[-1]


def _equal_time_positions(
    timestamps: list[int], points: list[tuple[float, float]], count: int,
) -> tuple[list[int], list[tuple[float, float]]]:
    start, end = timestamps[0], timestamps[-1]
    duration = end - start
    if duration <= 0:
        raise PreprocessingError("movement duration must be positive")
    sampled_timestamps = [start + round(duration * index / (count - 1)) for index in range(count)]
    sampled_timestamps[0], sampled_timestamps[-1] = start, end
    return sampled_timestamps, [_interpolate_position(timestamps, points, timestamp) for timestamp in sampled_timestamps]


def _resampling_error(
    source_timestamps: list[int], source_points: list[tuple[float, float]],
    sampled_timestamps: list[int], sampled_points: list[tuple[float, float]],
) -> float:
    return max(
        math.dist(point, _interpolate_position(sampled_timestamps, sampled_points, timestamp))
        for timestamp, point in zip(source_timestamps, source_points, strict=True)
    )


def _load_session_events(root: Path, manifest: dict[str, Any]) -> dict[str, list[tuple[int, int, int]]]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise PreprocessingError("PyArrow is required for preprocessing") from exc
    raw_root = (root / "raw_sessions").resolve()
    result: dict[str, list[tuple[int, int, int]]] = {}
    for raw_file in manifest["raw_files"]:
        candidate = (raw_root / raw_file["relative_path"]).resolve()
        if not candidate.is_relative_to(raw_root) or not candidate.is_file():
            raise PreprocessingError(f"raw evidence is missing: {raw_file['relative_path']}")
        if _digest_file(candidate) != raw_file["sha256"]:
            raise PreprocessingError(f"raw evidence hash changed: {raw_file['relative_path']}")
        table = pq.read_table(candidate, columns=["timestamp_ns", "screen_x", "screen_y"])
        events = [(int(timestamp), int(x), int(y)) for timestamp, x, y in zip(
            table.column("timestamp_ns").to_pylist(), table.column("screen_x").to_pylist(), table.column("screen_y").to_pylist(), strict=True
        )]
        result.setdefault(raw_file["session_id"], []).extend(events)
    for events in result.values():
        events.sort()
    return result


def _trial_rows(conn, snapshot_id: str) -> list[dict[str, Any]]:
    rows = conn.execute(
        """SELECT st.trial_id,st.session_id,st.split,st.ordinal,td.condition_json,td.target_appeared_ns,
                  td.valid_click_ns,td.clicks_json,td.start_screen_x,td.start_screen_y
             FROM dataset_snapshot_trials st
             JOIN trial_details td ON td.trial_id=st.trial_id
             WHERE st.snapshot_id=? ORDER BY st.ordinal""", (snapshot_id,)
    ).fetchall()
    return [dict(row) for row in rows]


def preprocess_dataset_snapshot(
    root: Path, database: Path, snapshot_id: str, spec: PreprocessingSpec = PreprocessingSpec(),
) -> dict[str, Any]:
    """Write a deterministic, immutable-source Parquet representation and reconstruction report."""
    conn = connect(database)
    run_id = ""
    run_dir: Path | None = None
    try:
        migrate(conn)
        repos = Repositories(conn)
        snapshot = repos.dataset_snapshot(snapshot_id)
        if snapshot["status"] != "ready" or not snapshot["manifest_relative_path"]:
            raise PreprocessingError("preprocessing requires a ready snapshot with a manifest")
        manifest_path = (root / snapshot["manifest_relative_path"]).resolve()
        if not manifest_path.is_file():
            raise PreprocessingError("snapshot manifest is missing")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if hashlib.sha256(_canonical_json(manifest).encode("utf-8")).hexdigest() != snapshot["manifest_sha256"]:
            raise PreprocessingError("snapshot manifest hash changed")
        config = spec.config()
        run_id = repos.create_preprocessing_run(snapshot_id, config, current_code_revision())
        repos.start_preprocessing_run(run_id)
        session_events = _load_session_events(root, manifest)
        rows = _trial_rows(conn, snapshot_id)
        if not rows:
            raise PreprocessingError("snapshot contains no trials")
        if any(
            json.loads(row["condition_json"]).get("collection_protocol_version") != 3
            or json.loads(row["condition_json"]).get("target_sampling_strategy") != "continuous_uniform_feasible_v3"
            for row in rows
        ):
            raise PreprocessingError("preprocessing schema 2 requires protocol-3 continuous-uniform trials")
        records: list[dict[str, Any]] = []
        skipped: list[dict[str, str]] = []
        resampling_errors: list[float] = []
        by_split: Counter[str] = Counter()
        for row in rows:
            try:
                if row["start_screen_x"] is None or row["start_screen_y"] is None or row["valid_click_ns"] is None:
                    raise PreprocessingError("trial lacks recorded start or valid-click timestamp")
                condition = json.loads(row["condition_json"])
                target = (float(condition["target_x"]), float(condition["target_y"]))
                start = (int(row["start_screen_x"]), int(row["start_screen_y"]))
                click = _valid_click(row["clicks_json"])
                timestamps, screen_points = _strict_points(
                    int(row["target_appeared_ns"]), start, click, session_events.get(row["session_id"], []),
                )
                onset_index = _movement_onset_index(screen_points, spec.onset_displacement_px)
                movement_timestamps, movement_points = timestamps[onset_index:], screen_points[onset_index:]
                sampled_timestamps, sampled_screen_points = _equal_time_positions(
                    movement_timestamps, movement_points, spec.equal_time_position_count,
                )
                transform = CanonicalTransform.from_start_target(
                    movement_points[0], target, float(condition["radius_px"]),
                )
                canonical_source_points = [transform.forward(point) for point in movement_points]
                canonical_positions = [transform.forward(point) for point in sampled_screen_points]
                canonical_positions[0] = (0.0, 0.0)
                resampling_error = _resampling_error(
                    movement_timestamps, canonical_source_points, sampled_timestamps, canonical_positions,
                )
                duration_ns = sampled_timestamps[-1] - sampled_timestamps[0]
                records.append({
                    "trial_id": row["trial_id"], "session_id": row["session_id"], "split": row["split"], "ordinal": row["ordinal"],
                    "schema_version": spec.schema_version, "movement_onset_ns": sampled_timestamps[0],
                    "total_movement_duration_ns": duration_ns, "log_total_movement_duration": math.log(duration_ns),
                    "source_point_count": len(movement_points),
                    "canonical_positions": [[point[0], point[1]] for point in canonical_positions],
                    "resampling_max_error": resampling_error,
                })
                resampling_errors.append(resampling_error)
                by_split[row["split"]] += 1
            except (KeyError, ValueError, PreprocessingError) as exc:
                skipped.append({"trial_id": row["trial_id"], "reason": str(exc)})
        if not records:
            raise PreprocessingError("no trials could be represented")
        run_dir = root / "datasets" / snapshot_id / "preprocessing" / run_id
        run_dir.mkdir(parents=True, exist_ok=False)
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
            processed_path = run_dir / "processed.parquet"
            pq.write_table(pa.Table.from_pylist(records), processed_path, compression="zstd")
        except ImportError as exc:
            raise PreprocessingError("PyArrow is required for preprocessing") from exc
        report = {
            "schema_version": spec.schema_version, "snapshot_id": snapshot_id, "run_id": run_id, "config": config,
            "processed_trial_count": len(records), "skipped_trial_count": len(skipped), "split_counts": dict(sorted(by_split.items())),
            "resampling": {"max_error": max(resampling_errors), "mean_error": sum(resampling_errors) / len(resampling_errors)},
            "skipped_trials": skipped,
        }
        report_path = run_dir / "reconstruction_report.json"
        report_path.write_text(_canonical_json(report) + "\n", encoding="utf-8")
        processed_relative = processed_path.relative_to(root).as_posix()
        report_relative = report_path.relative_to(root).as_posix()
        repos.complete_preprocessing_run(
            run_id, processed_relative, _digest_file(processed_path), report_relative, _digest_file(report_path), len(records), len(skipped),
        )
        return {"run_id": run_id, "processed_path": processed_relative, "report_path": report_relative, **report}
    except Exception as exc:
        if run_id:
            Repositories(conn).fail_preprocessing_run(run_id, str(exc))
        if run_dir is not None and run_dir.is_dir():
            shutil.rmtree(run_dir, ignore_errors=True)
        if isinstance(exc, PreprocessingError):
            raise
        raise PreprocessingError(str(exc)) from exc
    finally:
        conn.close()
