"""MVVM controller for small, metadata-only dataset snapshot construction."""
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QObject, Property, Signal, Slot

from mouselearn.datasets.snapshots import DatasetBuildError, build_dataset_snapshot
from mouselearn.domain.dataset import DatasetSnapshotPlan
from mouselearn.storage.database import connect, migrate
from mouselearn.storage.repositories import Repositories


class DatasetController(QObject):
    snapshotsChanged = Signal()
    messageChanged = Signal()

    def __init__(self, root: Path, database: Path, parent: QObject | None = None):
        super().__init__(parent)
        self.root, self.database = root, database
        self._snapshots: list[dict] = []
        self._message = "Build an immutable snapshot from retained completed sessions."
        self.refresh()

    @Property("QVariantList", notify=snapshotsChanged)
    def snapshots(self) -> list[dict]:
        return self._snapshots

    @Property(str, notify=messageChanged)
    def message(self) -> str:
        return self._message

    @Slot()
    def refresh(self) -> None:
        conn = connect(self.database)
        try:
            migrate(conn)
            self._snapshots = Repositories(conn).dataset_snapshots()
        finally:
            conn.close()
        self.snapshotsChanged.emit()

    @Slot(str)
    def buildSnapshot(self, name: str) -> None:
        try:
            snapshot = build_dataset_snapshot(self.root, self.database, DatasetSnapshotPlan(name=name.strip() or "Dataset snapshot"))
            self._message = (
                f"Snapshot {snapshot['id'][:8]} is ready: {snapshot['trial_count']} trials from "
                f"{snapshot['session_count']} session(s)."
            )
        except (DatasetBuildError, ValueError) as exc:
            self._message = f"Could not build snapshot: {exc}"
        self.messageChanged.emit()
        self.refresh()
