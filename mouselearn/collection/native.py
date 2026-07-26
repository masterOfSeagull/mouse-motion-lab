"""ctypes adapter for the Qt/Python-independent Windows Raw Input library."""
from __future__ import annotations

import ctypes
import os
import sys
from dataclasses import dataclass
from pathlib import Path


MML_CAPTURE_OK = 0
MML_CAPTURE_NOT_ACTIVE = 1
MML_CAPTURE_INVALID_ARGUMENT = 2
MML_CAPTURE_REGISTER_FAILED = 3
MML_CAPTURE_RAW_INPUT_FAILED = 4
MML_CAPTURE_BUFFER_OVERFLOW = 5
WM_INPUT = 0x00FF


class NativeMouseEvent(ctypes.Structure):
    _pack_ = 1
    _fields_ = [
        ("timestamp_ticks", ctypes.c_uint64),
        ("raw_dx", ctypes.c_int32),
        ("raw_dy", ctypes.c_int32),
        ("screen_x", ctypes.c_int32),
        ("screen_y", ctypes.c_int32),
        ("button_flags", ctypes.c_uint16),
        ("event_flags", ctypes.c_uint16),
        ("device_handle", ctypes.c_uint64),
        ("foreground_collection_window", ctypes.c_uint32),
        ("reserved", ctypes.c_uint32),
    ]


class NativeCaptureStats(ctypes.Structure):
    _fields_ = [
        ("active", ctypes.c_uint32),
        ("capacity", ctypes.c_uint32),
        ("buffered_events", ctypes.c_uint32),
        ("reserved", ctypes.c_uint32),
        ("observed_events", ctypes.c_uint64),
        ("overflow_events", ctypes.c_uint64),
        ("qpc_frequency_hz", ctypes.c_uint64),
        ("last_status", ctypes.c_uint32),
        ("reserved2", ctypes.c_uint32),
    ]


@dataclass(frozen=True)
class CaptureStats:
    active: bool
    capacity: int
    buffered_events: int
    observed_events: int
    overflow_events: int
    qpc_frequency_hz: int
    last_status: int


class NativeCaptureError(RuntimeError):
    pass


def native_library_path() -> Path | None:
    override = os.environ.get("MOUSE_MOTION_LAB_NATIVE_MOUSE_IO")
    if override:
        path = Path(override).expanduser().resolve()
        return path if path.is_file() else None
    if sys.platform != "win32":
        return None
    project_root = Path(__file__).resolve().parents[2]
    for path in (
        project_root / "build" / "native" / "Release" / "mousemotionlab_mouse_io.dll",
        project_root / "build" / "native" / "Debug" / "mousemotionlab_mouse_io.dll",
        Path(sys.executable).resolve().parent / "mousemotionlab_mouse_io.dll",
    ):
        if path.is_file():
            return path
    return None


class NativeMouseCapture:
    """Owns native registration; callers forward WM_INPUT and drain in small batches."""

    def __init__(self, library_path: Path | None = None):
        path = library_path or native_library_path()
        if path is None:
            raise NativeCaptureError("mousemotionlab_mouse_io.dll was not found; run tools/build-native.ps1")
        self.path = path
        self._dll = ctypes.WinDLL(str(path))
        self._dll.mml_capture_start.argtypes = [ctypes.c_size_t, ctypes.c_uint32]
        self._dll.mml_capture_start.restype = ctypes.c_uint32
        self._dll.mml_capture_stop.argtypes = []
        self._dll.mml_capture_stop.restype = ctypes.c_uint32
        self._dll.mml_capture_handle_raw_input.argtypes = [ctypes.c_ssize_t]
        self._dll.mml_capture_handle_raw_input.restype = ctypes.c_uint32
        self._dll.mml_capture_drain.argtypes = [ctypes.POINTER(NativeMouseEvent), ctypes.c_uint32, ctypes.POINTER(ctypes.c_uint32)]
        self._dll.mml_capture_drain.restype = ctypes.c_uint32
        self._dll.mml_capture_get_stats.argtypes = [ctypes.POINTER(NativeCaptureStats)]
        self._dll.mml_capture_get_stats.restype = ctypes.c_uint32
        self._dll.mml_capture_qpc_now.argtypes = []
        self._dll.mml_capture_qpc_now.restype = ctypes.c_uint64
        self._dll.mml_capture_cursor_position.argtypes = [ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_int32)]
        self._dll.mml_capture_cursor_position.restype = ctypes.c_uint32
        self._dll.mml_capture_client_to_screen.argtypes = [ctypes.c_size_t, ctypes.c_int32, ctypes.c_int32, ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_int32)]
        self._dll.mml_capture_client_to_screen.restype = ctypes.c_uint32

    def start(self, window_handle: int, capacity: int) -> None:
        status = self._dll.mml_capture_start(window_handle, capacity)
        if status != MML_CAPTURE_OK:
            raise NativeCaptureError(f"native capture start failed with status {status}")

    def stop(self) -> None:
        status = self._dll.mml_capture_stop()
        if status not in {MML_CAPTURE_OK, MML_CAPTURE_NOT_ACTIVE}:
            raise NativeCaptureError(f"native capture stop failed with status {status}")

    def handle_raw_input(self, raw_input_handle: int) -> int:
        return int(self._dll.mml_capture_handle_raw_input(raw_input_handle))

    def drain(self, limit: int = 2048) -> tuple[int, list[NativeMouseEvent]]:
        if limit <= 0:
            raise ValueError("drain limit must be positive")
        destination = (NativeMouseEvent * limit)()
        drained = ctypes.c_uint32()
        status = int(self._dll.mml_capture_drain(destination, limit, ctypes.byref(drained)))
        return status, list(destination[: drained.value])

    def stats(self) -> CaptureStats:
        stats = NativeCaptureStats()
        status = self._dll.mml_capture_get_stats(ctypes.byref(stats))
        if status != MML_CAPTURE_OK:
            raise NativeCaptureError(f"native capture stats failed with status {status}")
        return CaptureStats(
            active=bool(stats.active), capacity=int(stats.capacity), buffered_events=int(stats.buffered_events),
            observed_events=int(stats.observed_events), overflow_events=int(stats.overflow_events),
            qpc_frequency_hz=int(stats.qpc_frequency_hz), last_status=int(stats.last_status),
        )

    def qpc_now(self) -> int:
        return int(self._dll.mml_capture_qpc_now())

    def cursor_position(self) -> tuple[int, int]:
        screen_x, screen_y = ctypes.c_int32(), ctypes.c_int32()
        status = self._dll.mml_capture_cursor_position(ctypes.byref(screen_x), ctypes.byref(screen_y))
        if status != MML_CAPTURE_OK:
            raise NativeCaptureError(f"cursor position query failed with status {status}")
        return int(screen_x.value), int(screen_y.value)

    def client_to_screen(self, window_handle: int, client_x: int, client_y: int) -> tuple[int, int]:
        screen_x, screen_y = ctypes.c_int32(), ctypes.c_int32()
        status = self._dll.mml_capture_client_to_screen(window_handle, client_x, client_y, ctypes.byref(screen_x), ctypes.byref(screen_y))
        if status != MML_CAPTURE_OK:
            raise NativeCaptureError(f"coordinate conversion failed with status {status}")
        return int(screen_x.value), int(screen_y.value)
