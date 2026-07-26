"""Read one recorded trial's target-visible movement from session Parquet files."""
from __future__ import annotations

from pathlib import Path
from typing import Any


class TrajectoryLoadError(RuntimeError):
    pass


def load_trial_trajectory(raw_sessions_root: Path, trial: dict[str, Any], raw_files: list[dict[str, Any]], maximum_points: int = 700) -> dict[str, Any]:
    """Return display-ready coordinates limited to the trial's explicit phase bounds."""
    if maximum_points < 2:
        raise ValueError("maximum_points must be at least two")
    start_ns = trial.get("target_appeared_ns")
    end_ns = trial.get("valid_click_ns") or trial.get("ended_ns")
    if start_ns is None or end_ns is None:
        raise TrajectoryLoadError("trial has no completed target-visible interval")
    if end_ns < start_ns:
        raise TrajectoryLoadError("trial has invalid phase bounds")

    points: list[dict[str, int]] = []
    root = raw_sessions_root.resolve()
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise TrajectoryLoadError("PyArrow is required to read recorded trajectories") from exc

    for raw_file in raw_files:
        candidate = (root / raw_file["relative_path"]).resolve()
        if not candidate.is_relative_to(root):
            raise TrajectoryLoadError("raw event reference escapes the session data root")
        if not candidate.is_file():
            raise TrajectoryLoadError(f"recorded raw event file is missing: {raw_file['relative_path']}")
        table = pq.read_table(candidate, columns=["timestamp_ns", "screen_x", "screen_y"])
        for timestamp_ns, screen_x, screen_y in zip(
            table.column("timestamp_ns").to_pylist(), table.column("screen_x").to_pylist(), table.column("screen_y").to_pylist(), strict=True,
        ):
            if start_ns <= timestamp_ns <= end_ns:
                points.append({"timestamp_ns": int(timestamp_ns), "x": int(screen_x), "y": int(screen_y)})
    points.sort(key=lambda point: point["timestamp_ns"])
    raw_point_count = len(points)
    if len(points) > maximum_points:
        step = (len(points) - 1) / (maximum_points - 1)
        points = [points[round(index * step)] for index in range(maximum_points)]

    condition = trial["condition"]
    start = {"x": int(trial["start_screen_x"]), "y": int(trial["start_screen_y"])}
    target = {"x": int(condition["target_x"]), "y": int(condition["target_y"]), "radius": float(condition["radius_px"])}
    bounds_points = [start, {"x": target["x"], "y": target["y"]}, *points]
    min_x = min(point["x"] for point in bounds_points)
    max_x = max(point["x"] for point in bounds_points)
    min_y = min(point["y"] for point in bounds_points)
    max_y = max(point["y"] for point in bounds_points)
    padding = max(30, int(target["radius"]) * 2)
    return {
        "trial_id": trial["id"], "start": start, "target": target, "points": points,
        "raw_point_count": raw_point_count, "displayed_point_count": len(points),
        "duration_ms": round((end_ns - start_ns) / 1_000_000, 2),
        "bounds": {"min_x": min_x - padding, "max_x": max_x + padding, "min_y": min_y - padding, "max_y": max_y + padding},
    }
