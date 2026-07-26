"""Explicitly armed, local-canvas-only playback scheduler."""
from __future__ import annotations

import math
import threading
import time

from PySide6.QtCore import QObject, Property, QThread, QTimer, Qt, Signal, Slot
from PySide6.QtGui import QCursor, QGuiApplication


_generated_playback = threading.Event()


def generated_playback_active() -> bool:
    """Collection backends can use this marker to exclude generated activity."""
    return _generated_playback.is_set()


class _PlaybackThread(QThread):
    sampleReady = Signal(float, float, float)

    def __init__(self, points: list[dict], parent: QObject | None = None):
        super().__init__(parent)
        self.points = points

    def run(self) -> None:
        started = time.perf_counter_ns()
        total = max(1, int(self.points[-1]["time_ns"]))
        for point in self.points:
            target = started + int(point["time_ns"])
            while not self.isInterruptionRequested():
                remaining = target - time.perf_counter_ns()
                if remaining <= 0:
                    break
                if remaining > 2_000_000:
                    time.sleep((remaining - 1_000_000) / 1_000_000_000)
                else:
                    time.sleep(0)
            if self.isInterruptionRequested():
                return
            self.sampleReady.emit(float(point["x"]), float(point["y"]), min(1.0, int(point["time_ns"]) / total))


class PlaybackController(QObject):
    stateChanged = Signal()
    positionChanged = Signal()
    messageChanged = Signal()

    def __init__(self, parent: QObject | None = None):
        super().__init__(parent)
        self._state = "disarmed"
        self._message = "Playback is disarmed. Preview remains data-only."
        self._trajectory: list[dict] = []
        self._x = self._y = self._progress = 0.0
        self._thread: _PlaybackThread | None = None
        self._physical_anchor: tuple[int, int] | None = None
        self._monitor = QTimer(self); self._monitor.setInterval(20); self._monitor.timeout.connect(self._monitor_physical_mouse)
        app = QGuiApplication.instance()
        if app is not None:
            app.applicationStateChanged.connect(self._application_state_changed)

    @Property(str, notify=stateChanged)
    def state(self) -> str: return self._state

    @Property(str, notify=messageChanged)
    def message(self) -> str: return self._message

    @Property(float, notify=positionChanged)
    def x(self) -> float: return self._x

    @Property(float, notify=positionChanged)
    def y(self) -> float: return self._y

    @Property(float, notify=positionChanged)
    def progress(self) -> float: return self._progress

    @Property(bool, notify=stateChanged)
    def generatedInputActive(self) -> bool: return generated_playback_active()

    def setTrajectory(self, trajectory: dict) -> None:
        self._trajectory = list(trajectory.get("points", [])) if trajectory else []
        if self._state == "armed":
            self.disarm()

    def _set_state(self, state: str, message: str) -> None:
        self._state, self._message = state, message
        self.stateChanged.emit(); self.messageChanged.emit()

    @Slot()
    def arm(self) -> None:
        if self._state == "playing": return
        if len(self._trajectory) < 2:
            self._set_state("disarmed", "Generate a trajectory before arming playback.")
            return
        self._set_state("armed", "ARMED for one local-canvas playback. Start requires another explicit click.")

    @Slot()
    def disarm(self) -> None:
        if self._state == "playing": self.abort("Disarmed by user")
        else: self._set_state("disarmed", "Playback is disarmed.")

    @Slot()
    def start(self) -> None:
        if self._state != "armed":
            self._set_state("disarmed", "Playback did not start because it was not armed.")
            return
        self._x, self._y = float(self._trajectory[0]["x"]), float(self._trajectory[0]["y"])
        self._progress = 0.0; self.positionChanged.emit()
        cursor = QCursor.pos(); self._physical_anchor = (cursor.x(), cursor.y())
        _generated_playback.set()
        self._thread = _PlaybackThread(self._trajectory, self)
        self._thread.sampleReady.connect(self._sample, Qt.ConnectionType.QueuedConnection)
        self._thread.finished.connect(self._finished)
        self._set_state("playing", "Local playback active. Move the physical mouse or press Escape to abort.")
        self._monitor.start(); self._thread.start(QThread.Priority.TimeCriticalPriority)

    @Slot()
    def abortFromUi(self) -> None: self.abort("Emergency stop: Escape or Abort pressed")

    def abort(self, reason: str) -> None:
        if self._thread and self._thread.isRunning():
            self._thread.requestInterruption()
        _generated_playback.clear(); self._monitor.stop()
        self._set_state("disarmed", reason)

    @Slot(float, float, float)
    def _sample(self, x: float, y: float, progress: float) -> None:
        if self._state != "playing": return
        self._x, self._y, self._progress = x, y, progress; self.positionChanged.emit()

    def _finished(self) -> None:
        completed = self._state == "playing"
        _generated_playback.clear(); self._monitor.stop(); self._thread = None
        if completed:
            self._set_state("disarmed", "Local playback completed and automatically disarmed.")

    def _monitor_physical_mouse(self) -> None:
        if self._state != "playing" or self._physical_anchor is None: return
        cursor = QCursor.pos()
        self.notifyPhysicalDelta(cursor.x() - self._physical_anchor[0], cursor.y() - self._physical_anchor[1])

    def notifyPhysicalDelta(self, dx: float, dy: float) -> None:
        """Native/polling adapters and tests share the same substantial-movement gate."""
        if self._state == "playing" and math.hypot(dx, dy) >= 12:
            self.abort("Playback aborted by physical mouse movement.")

    def _application_state_changed(self, state: Qt.ApplicationState) -> None:
        if self._state == "playing" and state != Qt.ApplicationState.ApplicationActive:
            self.abort("Playback aborted because MouseMotionLab lost focus.")

    def shutdown(self) -> None:
        self.abort("Playback stopped during shutdown.")
        if self._thread:
            self._thread.wait(1000)


__all__ = ["PlaybackController", "generated_playback_active"]
