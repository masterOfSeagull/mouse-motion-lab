"""Experiment dashboard controller; all optimization remains in the worker."""
from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from PySide6.QtCore import QObject, Property, Signal, Slot

from mouselearn.storage.database import connect, migrate
from mouselearn.storage.repositories import Repositories


class TrainingController(QObject):
    experimentsChanged = Signal()
    messageChanged = Signal()

    def __init__(self, database: Path, start_training: Callable[[str, str], None], parent: QObject | None = None):
        super().__init__(parent)
        self.database, self._start_training = database, start_training
        self._experiments: list[dict] = []
        self._message = "Choose a completed representation and a conditional-flow preset."
        self.refresh()

    @Property("QVariantList", notify=experimentsChanged)
    def experiments(self) -> list[dict]:
        return self._experiments

    @Property(str, notify=messageChanged)
    def message(self) -> str:
        return self._message

    @Slot()
    def refresh(self) -> None:
        conn = connect(self.database)
        try:
            migrate(conn)
            self._experiments = Repositories(conn).experiments()
        finally:
            conn.close()
        self.experimentsChanged.emit()

    @Slot(str, str)
    def startTraining(self, preprocessing_run_id: str, preset: str) -> None:
        if not preprocessing_run_id or preset not in {"small", "standard"}:
            self._message = "Select a completed representation and valid preset."
        else:
            self._start_training(preprocessing_run_id, preset)
            self._message = f"Conditional-flow {preset} training is starting in the worker."
        self.messageChanged.emit()


__all__ = ["TrainingController"]
