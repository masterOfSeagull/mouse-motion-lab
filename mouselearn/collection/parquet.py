"""Bounded background writer for raw capture batches."""
from __future__ import annotations

import hashlib
import queue
import threading
import time
from pathlib import Path
from typing import Iterable

from mouselearn.domain.collection import RawEventFileReference

from .native import NativeMouseEvent


class CollectionPersistenceError(RuntimeError):
    pass


class BoundedParquetWriter:
    """Write raw events off the GUI thread; queue exhaustion is a visible failure."""

    def __init__(
        self, raw_sessions_root: Path, session_id: str, qpc_frequency_hz: int, queue_capacity: int = 64,
        row_group_event_count: int = 4096, row_group_max_delay_seconds: float = 0.25,
    ):
        if queue_capacity < 1:
            raise ValueError("queue_capacity must be positive")
        if row_group_event_count < 1 or row_group_max_delay_seconds <= 0:
            raise ValueError("row group thresholds must be positive")
        self.session_id = session_id
        self.qpc_frequency_hz = qpc_frequency_hz
        self.path = raw_sessions_root / session_id / "events.parquet"
        self.row_group_event_count = row_group_event_count
        self.row_group_max_delay_seconds = row_group_max_delay_seconds
        self._queue: queue.Queue[tuple[NativeMouseEvent, ...] | None] = queue.Queue(maxsize=queue_capacity)
        self._thread: threading.Thread | None = None
        self._failure: Exception | None = None
        self._event_count = 0
        self._first_timestamp_ns: int | None = None
        self._last_timestamp_ns: int | None = None

    @property
    def failure(self) -> Exception | None:
        return self._failure

    def start(self) -> None:
        self._require_pyarrow()
        if self._thread is not None:
            raise RuntimeError("writer already started")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._thread = threading.Thread(target=self._write, name=f"raw-event-writer-{self.session_id}", daemon=True)
        self._thread.start()

    def submit(self, events: Iterable[NativeMouseEvent]) -> bool:
        batch = tuple(events)
        if not batch:
            return True
        if self._failure is not None:
            return False
        try:
            self._queue.put_nowait(batch)
            return True
        except queue.Full:
            return False

    def finalize(self, timeout_seconds: float = 10.0) -> RawEventFileReference:
        if self._thread is None:
            raise RuntimeError("writer was not started")
        try:
            self._queue.put(None, timeout=timeout_seconds)
        except queue.Full as exc:
            raise CollectionPersistenceError("raw event writer did not drain before finalization") from exc
        self._thread.join(timeout_seconds)
        if self._thread.is_alive():
            raise CollectionPersistenceError("raw event writer did not finish")
        if self._failure is not None:
            raise CollectionPersistenceError(str(self._failure)) from self._failure
        digest = hashlib.sha256()
        with self.path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1_048_576), b""):
                digest.update(chunk)
        return RawEventFileReference(
            relative_path=f"{self.session_id}/events.parquet", event_count=self._event_count,
            first_timestamp_ns=self._first_timestamp_ns, last_timestamp_ns=self._last_timestamp_ns,
            qpc_frequency_hz=self.qpc_frequency_hz, byte_count=self.path.stat().st_size, sha256=digest.hexdigest(),
        )

    def _require_pyarrow(self) -> None:
        try:
            import pyarrow  # noqa: F401
            import pyarrow.parquet  # noqa: F401
        except ImportError as exc:
            raise CollectionPersistenceError("PyArrow is required for raw collection persistence") from exc

    def _write(self) -> None:
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq

            schema = pa.schema([
                ("timestamp_ns", pa.uint64()), ("raw_dx", pa.int32()), ("raw_dy", pa.int32()),
                ("screen_x", pa.int32()), ("screen_y", pa.int32()), ("button_flags", pa.uint16()),
                ("event_flags", pa.uint16()), ("device_handle", pa.uint64()),
                ("foreground_collection_window", pa.bool_()),
            ])
            with pq.ParquetWriter(self.path, schema, compression="zstd") as writer:
                pending: list[NativeMouseEvent] = []
                last_flush = time.monotonic()

                def flush_pending() -> None:
                    nonlocal pending, last_flush
                    if not pending:
                        return
                    table = pa.Table.from_pylist([
                        {
                            "timestamp_ns": event.timestamp_ticks * 1_000_000_000 // self.qpc_frequency_hz,
                            "raw_dx": event.raw_dx, "raw_dy": event.raw_dy,
                            "screen_x": event.screen_x, "screen_y": event.screen_y, "button_flags": event.button_flags,
                            "event_flags": event.event_flags, "device_handle": event.device_handle,
                            "foreground_collection_window": bool(event.foreground_collection_window),
                        }
                        for event in pending
                    ], schema=schema)
                    writer.write_table(table)
                    pending = []
                    last_flush = time.monotonic()

                while True:
                    timeout = max(0.001, self.row_group_max_delay_seconds - (time.monotonic() - last_flush))
                    try:
                        batch = self._queue.get(timeout=timeout)
                    except queue.Empty:
                        flush_pending()
                        continue
                    if batch is None:
                        break
                    pending.extend(batch)
                    self._event_count += len(batch)
                    first = batch[0].timestamp_ticks * 1_000_000_000 // self.qpc_frequency_hz
                    last = batch[-1].timestamp_ticks * 1_000_000_000 // self.qpc_frequency_hz
                    self._first_timestamp_ns = first if self._first_timestamp_ns is None else min(self._first_timestamp_ns, first)
                    self._last_timestamp_ns = last if self._last_timestamp_ns is None else max(self._last_timestamp_ns, last)
                    if len(pending) >= self.row_group_event_count:
                        flush_pending()
                flush_pending()
        except Exception as exc:  # propagated at the controlled finalize boundary
            self._failure = exc
