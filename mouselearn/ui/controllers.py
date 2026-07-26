from __future__ import annotations

import os
import sys
from pathlib import Path

from PySide6.QtCore import QObject, Property, QProcess, QProcessEnvironment, Signal, Slot

from mouselearn.collection.controller import CollectionController
from mouselearn.domain.events import WorkerEvent, parse_event
from mouselearn.storage.database import connect, migrate
from mouselearn.storage.repositories import Repositories
from mouselearn.ui.datasets import DatasetController
from mouselearn.ui.generator import GeneratorController
from mouselearn.ui.review import ReviewController
from mouselearn.ui.training import TrainingController
from mouselearn.ui.registry import RegistryController
from mouselearn.runtime import PlaybackController


class JobController(QObject):
    jobChanged = Signal()
    messageChanged = Signal()

    def __init__(self, database: Path, parent: QObject | None = None):
        super().__init__(parent)
        self.database = database
        self.process: QProcess | None = None
        self._job_id = ""
        self._message = "Ready"
        self._cancellation_requested = False

    @Property(str, notify=jobChanged)
    def jobId(self) -> str:
        return self._job_id

    @Property(str, notify=messageChanged)
    def message(self) -> str:
        return self._message

    def _set_message(self, text: str) -> None:
        self._message = text
        self.messageChanged.emit()

    def _repositories(self) -> tuple[object, Repositories]:
        conn = connect(self.database)
        migrate(conn)
        return conn, Repositories(conn)

    @Slot()
    def startDiagnostic(self) -> None:
        self._start_worker("diagnostic", ["diagnostic"], "Starting diagnostic worker…")

    @Slot(str)
    def startPreprocessing(self, snapshot_id: str) -> None:
        if not snapshot_id:
            self._set_message("Select a ready dataset snapshot first")
            return
        self._start_worker("preprocess", ["preprocess", "--snapshot-id", snapshot_id], "Starting preprocessing worker…")

    @Slot(str, str)
    def startBaseline(self, preprocessing_run_id: str, model_type: str) -> None:
        self._start_worker(
            "baseline", ["baseline", "--preprocessing-run-id", preprocessing_run_id, "--model-type", model_type],
            f"Starting {model_type.replace('_', ' ')} baseline worker…",
        )

    @Slot(str, str)
    def startFlowTraining(self, preprocessing_run_id: str, preset: str) -> None:
        self._start_worker(
            "conditional_flow", ["flow-train", "--preprocessing-run-id", preprocessing_run_id, "--preset", preset],
            f"Starting conditional-flow {preset} training worker…",
        )

    def _start_worker(self, job_type: str, command: list[str], message: str) -> None:
        if self.process and self.process.state() != QProcess.ProcessState.NotRunning:
            self._set_message("A job is already running")
            return
        conn, repos = self._repositories()
        try:
            self._job_id = repos.create_job(job_type)
        finally:
            conn.close()
        self.jobChanged.emit()
        self._cancellation_requested = False
        process = self.process = QProcess(self)
        process.setProgram(sys.executable)
        process.setArguments(["-m", "apps.worker", *command, "--job-id", self._job_id])
        environment = QProcessEnvironment.systemEnvironment()
        environment.insert("MOUSE_MOTION_LAB_DATA_ROOT", str(self.database.parent))
        environment.insert("PYTHONUNBUFFERED", "1")
        process.setProcessEnvironment(environment)
        process.readyReadStandardOutput.connect(self._read_stdout)
        process.readyReadStandardError.connect(self._read_stderr)
        process.finished.connect(self._finished)
        process.errorOccurred.connect(self._process_error)
        self._set_message(message)
        process.start()

    @Slot()
    def cancel(self) -> None:
        if self.process and self.process.state() != QProcess.ProcessState.NotRunning:
            self._cancellation_requested = True
            self.process.kill()
            self.process.waitForFinished(1500)
            self._set_job_terminal("cancelled", "Cancelled by user")
            conn, repos = self._repositories()
            try:
                repos.cancel_experiments_for_job(self._job_id)
            finally:
                conn.close()

    def _read_stdout(self) -> None:
        assert self.process is not None
        while self.process.canReadLine():
            line = bytes(self.process.readLine()).decode("utf-8", errors="replace").strip()
            if not line:
                continue
            try:
                event = parse_event(line)
                self._handle_event(event)
            except Exception as exc:
                self._set_job_terminal("failed", f"Malformed worker event: {exc}")

    def _read_stderr(self) -> None:
        assert self.process is not None
        text = bytes(self.process.readAllStandardError()).decode("utf-8", errors="replace").strip()
        if text:
            self._set_message(f"Worker: {text[-300:]}")

    def _handle_event(self, event: WorkerEvent) -> None:
        if event.job_id != self._job_id:
            self._set_job_terminal("failed", "Worker event job id does not match")
            return
        detail = event.message or event.stage or event.event
        self._set_message(detail)

    def _set_job_terminal(self, status: str, error: str | None = None) -> None:
        if not self._job_id:
            return
        conn, repos = self._repositories()
        try:
            job = repos.job(self._job_id)
            if job["status"] not in {"completed", "failed", "cancelled"}:
                repos.update_job(self._job_id, status=status, error=error)
        finally:
            conn.close()
        self._set_message(error or status)

    def _finished(self, exit_code: int, _exit_status: QProcess.ExitStatus) -> None:
        if self._cancellation_requested:
            self._set_job_terminal("cancelled", "Cancelled by user")
        elif exit_code != 0:
            self._set_job_terminal("failed", f"Worker exited with code {exit_code}")
        elif self._job_id:
            conn, repos = self._repositories()
            try:
                if repos.job(self._job_id)["status"] != "completed":
                    repos.update_job(self._job_id, status="failed", error="Worker ended without completion event")
            finally:
                conn.close()
        self.jobChanged.emit()

    def _process_error(self, _error: QProcess.ProcessError) -> None:
        if self.process:
            if self._cancellation_requested:
                self._set_job_terminal("cancelled", "Cancelled by user")
            else:
                self._set_job_terminal("failed", self.process.errorString())


class AppController(QObject):
    dashboardChanged = Signal()
    jobChanged = Signal()

    def __init__(self, root: Path, database: Path, schema_version: int):
        super().__init__()
        self.root, self.database, self.schema_version = root, database, schema_version
        self.jobs = JobController(database, self)
        self.collection = CollectionController(root, database, self)
        self.review = ReviewController(root, database, self)
        self.datasets = DatasetController(root, database, self.jobs.startPreprocessing, self)
        self.generator = GeneratorController(root, database, self.jobs.startBaseline, self)
        self.training = TrainingController(database, self.jobs.startFlowTraining, self)
        self.registry = RegistryController(root, database, self)
        self.playback = PlaybackController(self)
        self.generator.trajectoryChanged.connect(lambda: self.playback.setTrajectory(self.generator.trajectory))
        self.jobs.jobChanged.connect(self.refresh)
        self.jobs.jobChanged.connect(self.datasets.refresh)
        self.jobs.jobChanged.connect(self.generator.refresh)
        self.jobs.jobChanged.connect(self.training.refresh)
        self.jobs.jobChanged.connect(self.registry.refresh)
        self.jobs.messageChanged.connect(self.jobChanged)
        self.collection.stateChanged.connect(self.review.refresh)
        self.collection.stateChanged.connect(self.datasets.refresh)
        self._totals: dict[str, int] = {}
        self.refresh()

    @Property(str, constant=True)
    def dataRoot(self) -> str:
        return str(self.root)

    @Property(int, constant=True)
    def schemaVersion(self) -> int:
        return self.schema_version

    @Property(str, notify=dashboardChanged)
    def jobSummary(self) -> str:
        return ", ".join(f"{name}: {total}" for name, total in self._totals.items())

    @Property(str, notify=jobChanged)
    def jobMessage(self) -> str:
        return self.jobs.message

    @Property(QObject, constant=True)
    def collectionController(self) -> QObject:
        return self.collection

    @Property(QObject, constant=True)
    def reviewController(self) -> QObject:
        return self.review

    @Property(QObject, constant=True)
    def datasetController(self) -> QObject:
        return self.datasets

    @Property(QObject, constant=True)
    def generatorController(self) -> QObject:
        return self.generator

    @Property(QObject, constant=True)
    def trainingController(self) -> QObject:
        return self.training

    @Property(QObject, constant=True)
    def registryController(self) -> QObject:
        return self.registry

    @Property(QObject, constant=True)
    def playbackController(self) -> QObject:
        return self.playback

    @Slot()
    def refresh(self) -> None:
        conn = connect(self.database)
        try:
            self._totals = Repositories(conn).job_totals()
        finally:
            conn.close()
        self.dashboardChanged.emit()

    @Slot()
    def startDiagnostic(self) -> None:
        self.jobs.startDiagnostic()
        self.refresh()

    @Slot()
    def cancelJob(self) -> None:
        self.jobs.cancel()
        self.refresh()

    @Slot(QObject, int, int, int, int, int)
    def startCollection(self, window: QObject, planned_trials: int, canvas_x: int, canvas_y: int, canvas_width: int, canvas_height: int) -> None:
        if self.playback.state != "disarmed":
            self.playback.disarm()
        self.collection.start(window, planned_trials, canvas_x, canvas_y, canvas_width, canvas_height)

    @Slot()
    def stopCollection(self) -> None:
        self.collection.stop()

    @Slot()
    def startPlayback(self) -> None:
        if self.collection.state != "idle":
            self.playback.abort("Playback blocked while collection is active.")
            return
        self.playback.start()
