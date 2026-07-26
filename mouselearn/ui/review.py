"""MVVM review controller for logically excluding collection data from future datasets."""
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QObject, Property, Signal, Slot

from mouselearn.collection.trajectory import TrajectoryLoadError, load_trial_trajectory
from mouselearn.storage.database import connect, migrate
from mouselearn.storage.repositories import Repositories


class ReviewController(QObject):
    sessionsChanged = Signal()
    trialsChanged = Signal()
    messageChanged = Signal()
    selectedSessionChanged = Signal()
    trajectoryChanged = Signal()

    def __init__(self, root: Path, database: Path, parent: QObject | None = None):
        super().__init__(parent)
        self.root, self.database = root, database
        self._sessions: list[dict] = []
        self._trials: list[dict] = []
        self._selected_session_id = ""
        self._trajectory: dict = {}
        self._message = "Select a recorded session to review. Discarding is reversible and preserves raw files."
        self.refresh()

    def _repositories(self) -> tuple[object, Repositories]:
        conn = connect(self.database)
        migrate(conn)
        return conn, Repositories(conn)

    @Property("QVariantList", notify=sessionsChanged)
    def sessions(self) -> list[dict]:
        return self._sessions

    @Property("QVariantList", notify=trialsChanged)
    def trials(self) -> list[dict]:
        return self._trials

    @Property(str, notify=selectedSessionChanged)
    def selectedSessionId(self) -> str:
        return self._selected_session_id

    @Property(str, notify=messageChanged)
    def message(self) -> str:
        return self._message

    @Property("QVariantMap", notify=trajectoryChanged)
    def trajectory(self) -> dict:
        return self._trajectory

    @Slot()
    def refresh(self) -> None:
        conn, repos = self._repositories()
        try:
            self._sessions = repos.collection_sessions_for_review()
            if self._selected_session_id and any(row["id"] == self._selected_session_id for row in self._sessions):
                self._trials = repos.trials_for_review(self._selected_session_id)
            else:
                self._selected_session_id, self._trials = "", []
            self._trajectory = {}
        finally:
            conn.close()
        self.sessionsChanged.emit()
        self.trialsChanged.emit()
        self.selectedSessionChanged.emit()
        self.trajectoryChanged.emit()

    @Slot(str)
    def selectSession(self, session_id: str) -> None:
        conn, repos = self._repositories()
        try:
            repos.collection_session(session_id)
            self._selected_session_id = session_id
            self._trials = repos.trials_for_review(session_id)
            self._trajectory = {}
        except KeyError:
            self._message = "The selected session no longer exists."
        finally:
            conn.close()
        self.selectedSessionChanged.emit()
        self.trialsChanged.emit()
        self.trajectoryChanged.emit()

    @Slot(str)
    def selectTrial(self, trial_id: str) -> None:
        conn, repos = self._repositories()
        try:
            trial = repos.trial(trial_id)
            self._trajectory = load_trial_trajectory(self.root / "raw_sessions", trial, repos.raw_event_files(trial["session_id"]))
            self._message = f"Showing {self._trajectory['raw_point_count']} recorded points for the selected trial."
        except (KeyError, TrajectoryLoadError) as exc:
            self._trajectory = {}
            self._message = f"Could not load trajectory: {exc}"
        finally:
            conn.close()
        self.messageChanged.emit()
        self.trajectoryChanged.emit()

    @Slot(str)
    def discardSession(self, session_id: str) -> None:
        self._set_session_disposition(session_id, "discarded")

    @Slot(str)
    def retainSession(self, session_id: str) -> None:
        self._set_session_disposition(session_id, "retained")

    @Slot(str)
    def discardTrial(self, trial_id: str) -> None:
        self._set_trial_disposition(trial_id, "discarded")

    @Slot(str)
    def retainTrial(self, trial_id: str) -> None:
        self._set_trial_disposition(trial_id, "retained")

    def _set_session_disposition(self, session_id: str, disposition: str) -> None:
        conn, repos = self._repositories()
        try:
            repos.set_session_review(session_id, disposition)
            self._message = f"Session marked {disposition}. Raw files were retained."
        finally:
            conn.close()
        self.messageChanged.emit()
        self.refresh()

    def _set_trial_disposition(self, trial_id: str, disposition: str) -> None:
        conn, repos = self._repositories()
        try:
            repos.set_trial_review(trial_id, disposition)
            self._message = f"Trial marked {disposition}. Raw files were retained."
        finally:
            conn.close()
        self.messageChanged.emit()
        self.refresh()
