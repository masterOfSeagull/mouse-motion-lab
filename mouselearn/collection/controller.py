"""Qt-facing collection coordinator. QML only renders its exposed state."""
from __future__ import annotations

import ctypes
import importlib.util
from math import atan2, degrees, hypot, log2
import random
from pathlib import Path
from typing import Any

from PySide6.QtCore import QAbstractNativeEventFilter, QObject, Property, QTimer, Signal, Slot
from PySide6.QtGui import QGuiApplication

from mouselearn.domain.collection import (
    CaptureHealthRecord,
    ClickRecord,
    CollectionPhaseMarker,
    CollectionSessionPlan,
    TargetCondition,
    TrialFinalization,
    TrialPlan,
)
from mouselearn.storage.database import connect, migrate
from mouselearn.storage.repositories import Repositories

from .native import MML_CAPTURE_BUFFER_OVERFLOW, MML_CAPTURE_OK, NativeCaptureError, NativeMouseCapture, WM_INPUT, native_library_path
from .parquet import BoundedParquetWriter, CollectionPersistenceError
from .targets import BalancedTargetScheduler


RI_MOUSE_LEFT_BUTTON_DOWN = 0x0001


class _Point(ctypes.Structure):
    _fields_ = [("x", ctypes.c_long), ("y", ctypes.c_long)]


class _WindowsMessage(ctypes.Structure):
    _fields_ = [
        ("hwnd", ctypes.c_void_p), ("message", ctypes.c_uint), ("wParam", ctypes.c_size_t),
        ("lParam", ctypes.c_ssize_t), ("time", ctypes.c_uint32), ("pt", _Point), ("lPrivate", ctypes.c_uint32),
    ]


class _RawInputEventFilter(QAbstractNativeEventFilter):
    def __init__(self, capture: NativeMouseCapture):
        super().__init__()
        self.capture = capture
        self.failure_status: int | None = None

    def nativeEventFilter(self, _event_type: Any, message: Any) -> tuple[bool, int]:  # noqa: N802 - Qt API name
        pointer = int(message)
        if pointer:
            native_message = ctypes.cast(pointer, ctypes.POINTER(_WindowsMessage)).contents
            if native_message.message == WM_INPUT:
                status = self.capture.handle_raw_input(int(native_message.lParam))
                if status != MML_CAPTURE_OK:
                    self.failure_status = status
        return False, 0


class CollectionController(QObject):
    stateChanged = Signal()
    targetChanged = Signal()
    messageChanged = Signal()
    sessionChanged = Signal()
    captureChanged = Signal()

    def __init__(self, root: Path, database: Path, parent: QObject | None = None):
        super().__init__(parent)
        self.root, self.database = root, database
        self._state = "idle"
        self._message = self._availability_message()
        self._session_id = ""
        self._trial_id = ""
        self._planned_trials = 0
        self._completed_trials = 0
        self._target_visible = False
        self._target_x = 0
        self._target_y = 0
        self._target_radius = 0
        self._target_physical_x = 0
        self._target_physical_y = 0
        self._observed_events = 0
        self._buffered_events = 0
        self._overflow_events = 0
        self._qpc_frequency_hz = 0
        self._window: QObject | None = None
        self._canvas_x = 0
        self._canvas_y = 0
        self._canvas_width = 0
        self._canvas_height = 0
        self._capture: NativeMouseCapture | None = None
        self._filter: _RawInputEventFilter | None = None
        self._writer: BoundedParquetWriter | None = None
        self._clicks: list[ClickRecord] = []
        self._random = random.Random()
        self._scheduler: BalancedTargetScheduler | None = None
        self._pending_target: Any | None = None
        self._trial_token = 0
        self._active_trial_token = 0
        self._finishing = False
        self._frame_signal: Any | None = None
        self._timer = QTimer(self)
        self._timer.setInterval(8)
        self._timer.timeout.connect(self._drain_capture)
        self._trial_timeout = QTimer(self)
        self._trial_timeout.setSingleShot(True)
        self._trial_timeout.timeout.connect(self._on_trial_timeout)

    def _availability_message(self) -> str:
        if native_library_path() is None:
            return "Native capture library unavailable. Run tools/build-native.ps1."
        if importlib.util.find_spec("pyarrow") is None:
            return "PyArrow is unavailable. Run tools/setup-dev.ps1."
        return "Ready to collect. Raw Input activates only during an active session."

    @Property(bool, notify=stateChanged)
    def available(self) -> bool:
        return native_library_path() is not None and importlib.util.find_spec("pyarrow") is not None

    @Property(str, notify=stateChanged)
    def state(self) -> str:
        return self._state

    @Property(str, notify=messageChanged)
    def message(self) -> str:
        return self._message

    @Property(str, notify=sessionChanged)
    def sessionId(self) -> str:
        return self._session_id

    @Property(int, notify=stateChanged)
    def plannedTrials(self) -> int:
        return self._planned_trials

    @Property(int, notify=stateChanged)
    def completedTrials(self) -> int:
        return self._completed_trials

    @Property(bool, notify=targetChanged)
    def targetVisible(self) -> bool:
        return self._target_visible

    @Property(int, notify=targetChanged)
    def targetX(self) -> int:
        return self._target_x

    @Property(int, notify=targetChanged)
    def targetY(self) -> int:
        return self._target_y

    @Property(int, notify=targetChanged)
    def targetRadius(self) -> int:
        return self._target_radius

    @Property(int, notify=captureChanged)
    def observedEvents(self) -> int:
        return self._observed_events

    @Property(int, notify=captureChanged)
    def bufferedEvents(self) -> int:
        return self._buffered_events

    @Property(int, notify=captureChanged)
    def overflowEvents(self) -> int:
        return self._overflow_events

    @Property(int, notify=captureChanged)
    def qpcFrequencyHz(self) -> int:
        return self._qpc_frequency_hz

    def _repositories(self) -> tuple[object, Repositories]:
        conn = connect(self.database)
        migrate(conn)
        return conn, Repositories(conn)

    def _set_state(self, state: str, message: str) -> None:
        self._state, self._message = state, message
        self.stateChanged.emit()
        self.messageChanged.emit()

    @Slot(QObject, int, int, int, int, int)
    def start(self, window: QObject, planned_trials: int, canvas_x: int, canvas_y: int, canvas_width: int, canvas_height: int) -> None:
        if self._state != "idle":
            self._set_state(self._state, "A collection session is already active.")
            return
        if not self.available:
            self._set_state("idle", self._availability_message())
            return
        if planned_trials < 1 or planned_trials > 500:
            self._set_state("idle", "Choose between 1 and 500 trials.")
            return
        if canvas_width < 100 or canvas_height < 100:
            self._set_state("idle", "Collection canvas has not been laid out yet.")
            return
        try:
            window_handle = int(window.winId())
            self._capture = NativeMouseCapture()
            seed = random.SystemRandom().randrange(2**31)
            conn, repos = self._repositories()
            try:
                self._session_id = repos.create_collection_session(CollectionSessionPlan(
                    display_name="Qt Quick collection", mode="balanced_coverage", planned_trials=planned_trials,
                    random_seed=seed,
                    config={
                        "collection_protocol_version": 2, "inter_trial_delay_ms": [400, 1200],
                        "trial_timeout_ms": 15_000, "writer_queue_batches": 64, "parquet_row_group_events": 4096,
                    },
                    environment={"native_library": str(self._capture.path)},
                ))
            finally:
                conn.close()
            self._capture.start(window_handle, capacity=16_384)
            stats = self._capture.stats()
            self._update_capture_stats(stats)
            self._writer = BoundedParquetWriter(
                self.root / "raw_sessions", self._session_id, stats.qpc_frequency_hz, row_group_event_count=4096,
            )
            self._writer.start()
            conn, repos = self._repositories()
            try:
                repos.update_collection_environment(self._session_id, {
                    "qpc_frequency_hz": stats.qpc_frequency_hz,
                    "coordinate_system": "native physical pixels; Qt logical pixels converted through ClientToScreen",
                    "qt_device_pixel_ratio": float(window.devicePixelRatio()),
                })
                repos.transition_collection_session(self._session_id, "active")
                repos.set_collection_quality(self._session_id, "current", [])
            finally:
                conn.close()
            self._window = window
            self._canvas_x, self._canvas_y = canvas_x, canvas_y
            self._canvas_width, self._canvas_height = canvas_width, canvas_height
            self._planned_trials = planned_trials
            self._completed_trials = 0
            self._random.seed(seed)
            self._scheduler = BalancedTargetScheduler(seed)
            self._finishing = False
            self._filter = _RawInputEventFilter(self._capture)
            QGuiApplication.instance().installNativeEventFilter(self._filter)
            self._frame_signal = getattr(window, "frameSwapped", None)
            if self._frame_signal is None:
                raise RuntimeError("collection window cannot confirm frame presentation")
            self._frame_signal.connect(self._on_frame_swapped)
            self._timer.start()
            self._set_state("active", "Collection active. Raw Input is registered for this window only.")
            self._record_phase("inter_trial")
            self.sessionChanged.emit()
            self._schedule_next_trial(0)
        except (NativeCaptureError, CollectionPersistenceError, OSError, AttributeError) as exc:
            self._fail(f"Collection could not start: {exc}")

    @Slot()
    def stop(self) -> None:
        if self._state == "active":
            self._finish("cancelled", "Stopped by user")

    def _schedule_next_trial(self, delay_ms: int | None = None) -> None:
        if self._state != "active":
            return
        delay = delay_ms if delay_ms is not None else self._random.randint(400, 1200)
        QTimer.singleShot(delay, self._begin_trial)

    def _begin_trial(self) -> None:
        if self._state != "active" or self._window is None or self._capture is None:
            return
        width, height = self._canvas_width, self._canvas_height
        if self._scheduler is None:
            self._fail("Target scheduler is not initialized.")
            return
        try:
            cursor_x, cursor_y = self._cursor_canvas_position()
            target = self._scheduler.next(width, height, cursor_x, cursor_y)
        except (NativeCaptureError, ValueError) as exc:
            self._fail(f"Could not schedule cursor-relative target: {exc}")
            return
        radius = target.radius
        if width <= 2 * radius + 40 or height <= 2 * radius + 40:
            self._fail("Collection window is too small for a target.")
            return
        self._target_radius = radius
        self._target_x = target.x
        self._target_y = target.y
        dpr = float(self._window.devicePixelRatio())
        try:
            self._target_physical_x, self._target_physical_y = self._capture.client_to_screen(
                int(self._window.winId()), round((self._canvas_x + self._target_x) * dpr), round((self._canvas_y + self._target_y) * dpr),
            )
            self._target_visible = True
            self._pending_target = target
            self._targetChanged()
        except (NativeCaptureError, RuntimeError, ValueError) as exc:
            self._fail(f"Could not create trial: {exc}")
            return
        self._clicks = []

    def _cursor_canvas_position(self) -> tuple[float, float]:
        if self._capture is None or self._window is None:
            raise NativeCaptureError("native capture is unavailable")
        dpr = float(self._window.devicePixelRatio())
        origin_x, origin_y = self._capture.client_to_screen(
            int(self._window.winId()), round(self._canvas_x * dpr), round(self._canvas_y * dpr),
        )
        cursor_x, cursor_y = self._capture.cursor_position()
        return (cursor_x - origin_x) / dpr, (cursor_y - origin_y) / dpr

    def _on_frame_swapped(self) -> None:
        """Only now is the target visibly presented and eligible for reaction timing/clicks."""
        if self._state != "active" or self._finishing or self._pending_target is None or self._capture is None:
            return
        target = self._pending_target
        try:
            timestamp_ns, start_screen_x, start_screen_y = self._capture_timestamp_and_cursor()
            dpr = float(self._window.devicePixelRatio()) if self._window else 1.0
            distance = hypot(self._target_physical_x - start_screen_x, self._target_physical_y - start_screen_y)
            angle = degrees(atan2(self._target_physical_y - start_screen_y, self._target_physical_x - start_screen_x)) % 360
            difficulty = log2(distance / (2 * target.radius * dpr) + 1) if distance else 0.0
            condition = TargetCondition(
                distance_px=distance, radius_px=target.radius * dpr, angle_degrees=angle,
                screen_region=target.screen_region, difficulty_band=self._difficulty_band(difficulty),
                target_x=self._target_physical_x, target_y=self._target_physical_y, monitor_id="active-window",
                requested_distance_px=target.requested_distance_px * dpr,
                requested_radius_px=target.requested_radius_px * dpr,
                requested_angle_degrees=target.requested_angle_degrees,
                requested_screen_region=target.requested_screen_region, requested_corner=target.requested_corner,
                realized_corner=target.realized_corner, collection_protocol_version=2, reaction_time_confidence="high",
            )
            conn, repos = self._repositories()
            try:
                self._trial_id = repos.create_trial(TrialPlan(
                    session_id=self._session_id, condition=condition, target_appeared_ns=timestamp_ns,
                    start_screen_x=start_screen_x, start_screen_y=start_screen_y,
                ))
                repos.record_phase_marker(
                    self._session_id,
                    CollectionPhaseMarker(phase="target_visible", timestamp_ns=timestamp_ns, screen_x=start_screen_x, screen_y=start_screen_y),
                    self._trial_id,
                )
            finally:
                conn.close()
            self._pending_target = None
            self._trial_token += 1
            self._active_trial_token = self._trial_token
            self._trial_timeout.start(15_000)
        except (NativeCaptureError, RuntimeError, ValueError) as exc:
            self._fail(f"Could not activate presented target: {exc}")

    @staticmethod
    def _difficulty_band(index_of_difficulty: float) -> str:
        if index_of_difficulty < 2:
            return "low"
        if index_of_difficulty < 4:
            return "medium"
        return "high"

    def _targetChanged(self) -> None:
        self.targetChanged.emit()

    def _qpc_to_ns(self, ticks: int) -> int:
        if self._capture is None:
            return 0
        frequency = self._capture.stats().qpc_frequency_hz
        return ticks * 1_000_000_000 // frequency if frequency else 0

    def _capture_timestamp_and_cursor(self) -> tuple[int, int, int]:
        if self._capture is None:
            raise NativeCaptureError("native capture is unavailable")
        ticks = self._capture.qpc_now()
        screen_x, screen_y = self._capture.cursor_position()
        return self._qpc_to_ns(ticks), screen_x, screen_y

    def _record_phase(self, phase: str, trial_id: str | None = None, timestamp_ns: int | None = None, screen_x: int | None = None, screen_y: int | None = None) -> None:
        if not self._session_id:
            return
        if timestamp_ns is None:
            timestamp_ns, screen_x, screen_y = self._capture_timestamp_and_cursor()
        conn, repos = self._repositories()
        try:
            repos.record_phase_marker(
                self._session_id,
                CollectionPhaseMarker(phase=phase, timestamp_ns=timestamp_ns, screen_x=screen_x, screen_y=screen_y),
                trial_id,
            )
        finally:
            conn.close()

    def _drain_capture(self, process_actions: bool = True) -> None:
        if (self._state != "active" and not self._finishing) or self._capture is None or self._writer is None:
            return
        if self._filter and self._filter.failure_status is not None:
            self._fail(f"Raw Input handling failed with status {self._filter.failure_status}")
            return
        try:
            status, events = self._capture.drain()
            self._update_capture_stats(self._capture.stats())
            if status == MML_CAPTURE_BUFFER_OVERFLOW:
                self._fail("Raw Input buffer overflow; session stopped so no event loss is hidden.")
                return
            if status not in {MML_CAPTURE_OK}:
                self._fail(f"Raw Input drain failed with status {status}")
                return
            if events and not self._writer.submit(events):
                self._fail("Parquet writer queue overflow; session stopped so no event loss is hidden.")
                return
            if self._writer.failure is not None:
                self._fail(f"Parquet writer failed: {self._writer.failure}")
                return
            if process_actions and not self._finishing:
                for event in events:
                    if event.button_flags & RI_MOUSE_LEFT_BUTTON_DOWN:
                        self._record_click(event)
        except NativeCaptureError as exc:
            self._fail(str(exc))

    def _update_capture_stats(self, stats: Any) -> None:
        self._observed_events = stats.observed_events
        self._buffered_events = stats.buffered_events
        self._overflow_events = stats.overflow_events
        self._qpc_frequency_hz = stats.qpc_frequency_hz
        self.captureChanged.emit()

    def _record_click(self, event: Any) -> None:
        if not self._trial_id or not self._target_visible or self._capture is None:
            return
        dx, dy = event.screen_x - self._target_physical_x, event.screen_y - self._target_physical_y
        radius = self._target_radius * float(self._window.devicePixelRatio()) if self._window else 0
        valid = dx * dx + dy * dy <= radius * radius
        self._clicks.append(ClickRecord(
            timestamp_ns=self._qpc_to_ns(event.timestamp_ticks), screen_x=event.screen_x, screen_y=event.screen_y, is_valid=valid,
        ))
        if valid:
            timestamp_ns = self._qpc_to_ns(event.timestamp_ticks)
            self._finalize_active_trial("completed", "valid_click", "trial_completed", timestamp_ns, event.screen_x, event.screen_y)
            self._record_phase("inter_trial", timestamp_ns=timestamp_ns, screen_x=event.screen_x, screen_y=event.screen_y)
            self._completed_trials += 1
            self._target_visible = False
            self._targetChanged()
            self.stateChanged.emit()
            if self._completed_trials >= self._planned_trials:
                self._finish("completed", "Collection completed.")
            else:
                self._schedule_next_trial()

    def _on_trial_timeout(self) -> None:
        if self._state != "active" or self._finishing or not self._trial_id:
            return
        if self._active_trial_token != self._trial_token:
            return
        try:
            timestamp_ns, screen_x, screen_y = self._capture_timestamp_and_cursor()
            self._finalize_active_trial("failed", "timeout", "trial_timed_out", timestamp_ns, screen_x, screen_y)
            self._target_visible = False
            self._targetChanged()
            self._schedule_next_trial()
        except NativeCaptureError as exc:
            self._fail(f"Could not finalize timed out trial: {exc}")

    def _finalize_active_trial(
        self, state: str, end_reason: str, phase: str, timestamp_ns: int, screen_x: int, screen_y: int,
    ) -> None:
        if not self._trial_id:
            return
        trial_id = self._trial_id
        conn, repos = self._repositories()
        try:
            repos.finalize_trial(trial_id, TrialFinalization(
                state=state, end_reason=end_reason, ended_ns=timestamp_ns, clicks=tuple(self._clicks),
            ))
            repos.record_phase_marker(
                self._session_id,
                CollectionPhaseMarker(phase=phase, timestamp_ns=timestamp_ns, screen_x=screen_x, screen_y=screen_y),
                trial_id,
            )
        finally:
            conn.close()
        self._trial_id = ""
        self._trial_timeout.stop()

    def _fail(self, message: str) -> None:
        if self._session_id:
            conn, repos = self._repositories()
            try:
                repos.record_capture_health(self._session_id, CaptureHealthRecord(severity="error", code="capture_failure", occurred_at_ns=0, detail={"message": message}))
            finally:
                conn.close()
        self._finish("failed", message)

    def _finish(self, terminal_state: str, message: str) -> None:
        if self._finishing:
            return
        self._finishing = True
        self._timer.stop()
        self._trial_timeout.stop()
        self._target_visible = False
        self._pending_target = None
        self._targetChanged()
        if self._filter is not None:
            QGuiApplication.instance().removeNativeEventFilter(self._filter)
            self._filter = None
        if self._capture is not None:
            try:
                self._capture.stop()
            except NativeCaptureError:
                pass
        # Capture has been unregistered; drain its preallocated ring before finalizing any metadata.
        try:
            while self._capture is not None and self._writer is not None:
                status, events = self._capture.drain()
                self._update_capture_stats(self._capture.stats())
                if events and not self._writer.submit(events):
                    terminal_state, message = "failed", "Parquet writer queue overflow during final capture drain."
                    break
                if not events:
                    break
                if status not in {MML_CAPTURE_OK, MML_CAPTURE_BUFFER_OVERFLOW}:
                    terminal_state, message = "failed", f"Raw Input final drain failed with status {status}"
                    break
        except NativeCaptureError as exc:
            terminal_state, message = "failed", f"Raw Input final drain failed: {exc}"
        file_reference = None
        writer_error: Exception | None = None
        if self._writer is not None:
            try:
                file_reference = self._writer.finalize()
            except CollectionPersistenceError as exc:
                writer_error = exc
        if writer_error is not None:
            terminal_state, message = "failed", f"Raw event persistence failed: {writer_error}"
        if self._session_id:
            conn, repos = self._repositories()
            try:
                if self._trial_id:
                    try:
                        timestamp_ns, screen_x, screen_y = self._capture_timestamp_and_cursor()
                    except NativeCaptureError:
                        timestamp_ns, screen_x, screen_y = 0, 0, 0
                    if terminal_state == "cancelled":
                        self._finalize_active_trial("cancelled", "cancelled", "trial_cancelled", timestamp_ns, screen_x, screen_y)
                    else:
                        self._finalize_active_trial("failed", "capture_failure", "trial_failed", timestamp_ns, screen_x, screen_y)
                if file_reference is not None:
                    repos.record_raw_event_file(self._session_id, file_reference)
                phase = {
                    "completed": "session_completed", "cancelled": "session_cancelled", "failed": "session_failed",
                }[terminal_state]
                try:
                    timestamp_ns, screen_x, screen_y = self._capture_timestamp_and_cursor()
                except NativeCaptureError:
                    timestamp_ns, screen_x, screen_y = 0, 0, 0
                repos.record_phase_marker(
                    self._session_id,
                    CollectionPhaseMarker(phase=phase, timestamp_ns=timestamp_ns, screen_x=screen_x, screen_y=screen_y),
                )
                repos.transition_collection_session(self._session_id, terminal_state, message if terminal_state == "failed" else None)
            finally:
                conn.close()
        if self._frame_signal is not None:
            try:
                self._frame_signal.disconnect(self._on_frame_swapped)
            except (RuntimeError, TypeError):
                pass
            self._frame_signal = None
        self._capture = None
        self._writer = None
        self._trial_id = ""
        self._window = None
        self._canvas_width = 0
        self._canvas_height = 0
        self._finishing = False
        self._set_state("idle", message)
