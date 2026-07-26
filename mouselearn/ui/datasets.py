"""MVVM controller for small, metadata-only dataset snapshot construction."""
from __future__ import annotations

import json
from pathlib import Path
from collections.abc import Callable

from PySide6.QtCore import QObject, Property, Signal, Slot

from mouselearn.datasets.snapshots import DatasetBuildError, build_dataset_snapshot
from mouselearn.domain.dataset import DatasetSnapshotPlan
from mouselearn.storage.database import connect, migrate
from mouselearn.storage.repositories import Repositories


class DatasetController(QObject):
    snapshotsChanged = Signal()
    preprocessingRunsChanged = Signal()
    messageChanged = Signal()

    def __init__(self, root: Path, database: Path, start_preprocessing: Callable[[str], None], parent: QObject | None = None):
        super().__init__(parent)
        self.root, self.database = root, database
        self._start_preprocessing = start_preprocessing
        self._snapshots: list[dict] = []
        self._preprocessing_runs: list[dict] = []
        self._message = "Build an immutable snapshot from retained completed sessions."
        self.refresh()

    @Property("QVariantList", notify=snapshotsChanged)
    def snapshots(self) -> list[dict]:
        return self._snapshots

    @Property("QVariantList", notify=preprocessingRunsChanged)
    def preprocessingRuns(self) -> list[dict]:
        return self._preprocessing_runs

    @Property(str, notify=messageChanged)
    def message(self) -> str:
        return self._message

    @Slot()
    def refresh(self) -> None:
        conn = connect(self.database)
        try:
            migrate(conn)
            repositories = Repositories(conn)
            self._snapshots = repositories.dataset_snapshots()
            self._preprocessing_runs = repositories.preprocessing_runs()
            for run in self._preprocessing_runs:
                report_path = run.get("report_relative_path")
                if not report_path:
                    continue
                candidate = (self.root / report_path).resolve()
                if candidate.is_relative_to(self.root.resolve()) and candidate.is_file():
                    try:
                        report = json.loads(candidate.read_text(encoding="utf-8"))
                        run["resampling_max_error"] = report.get("resampling", {}).get("max_error")
                    except (OSError, json.JSONDecodeError):
                        run["resampling_max_error"] = None
        finally:
            conn.close()
        self.snapshotsChanged.emit()
        self.preprocessingRunsChanged.emit()

    @Slot(str)
    def buildSnapshot(self, name: str) -> None:
        try:
            conn = connect(self.database)
            try:
                current_sessions = Repositories(conn).current_protocol_dataset_sessions()
            finally:
                conn.close()
            if not current_sessions:
                raise ValueError("No retained protocol-3 sessions are eligible. Record a continuous-uniform protocol-3 session first.")
            snapshot = build_dataset_snapshot(
                self.root, self.database,
                DatasetSnapshotPlan(name=name.strip() or "Current protocol dataset", session_ids=tuple(row["id"] for row in current_sessions)),
            )
            self._message = (
                f"Snapshot {snapshot['id'][:8]} is ready: {snapshot['trial_count']} trials from "
                f"{snapshot['session_count']} protocol-3 session(s)."
            )
        except (DatasetBuildError, ValueError) as exc:
            self._message = f"Could not build snapshot: {exc}"
        self.messageChanged.emit()
        self.refresh()

    @Slot(str)
    def preprocessSnapshot(self, snapshot_id: str) -> None:
        try:
            conn = connect(self.database)
            try:
                snapshot = Repositories(conn).dataset_snapshot(snapshot_id)
            finally:
                conn.close()
            if snapshot["status"] != "ready":
                raise ValueError("Only ready snapshots can be preprocessed")
            if any(
                condition.get("collection_protocol_version") != 3
                or condition.get("target_sampling_strategy") != "continuous_uniform_feasible_v3"
                for condition in self._snapshot_conditions(snapshot["ordered_trial_ids"])
            ):
                raise ValueError("This snapshot is not protocol 3. Build a retained continuous-uniform protocol-3 snapshot first.")
            self._start_preprocessing(snapshot_id)
            self._message = f"Preprocessing {snapshot_id[:8]} in the background."
        except (KeyError, ValueError) as exc:
            self._message = f"Could not start preprocessing: {exc}"
        self.messageChanged.emit()

    def _snapshot_conditions(self, trial_ids: list[str]) -> list[dict]:
        if not trial_ids:
            return []
        conn = connect(self.database)
        try:
            placeholders = ",".join("?" for _ in trial_ids)
            rows = conn.execute(
                f"SELECT condition_json FROM trial_details WHERE trial_id IN ({placeholders})", trial_ids,
            ).fetchall()
            return [json.loads(row[0]) for row in rows]
        finally:
            conn.close()
