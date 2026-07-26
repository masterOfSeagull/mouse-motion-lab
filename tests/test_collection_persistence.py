from __future__ import annotations

import ctypes

import pyarrow.parquet as pq

from mouselearn.collection.native import NativeMouseEvent
from mouselearn.collection.parquet import BoundedParquetWriter


def _event(ticks: int, dx: int, flags: int = 0) -> NativeMouseEvent:
    return NativeMouseEvent(ticks, dx, -2, 100 + dx, 200, flags, 0, 123, 1, 0)


def test_native_event_abi_is_packed() -> None:
    assert ctypes.sizeof(NativeMouseEvent) == 44


def test_bounded_parquet_writer_converts_qpc_ticks_to_nanoseconds(tmp_path) -> None:
    writer = BoundedParquetWriter(tmp_path / "raw_sessions", "session-1", qpc_frequency_hz=10_000_000, queue_capacity=2)
    writer.start()
    assert writer.submit([_event(10_000_000, 4), _event(15_000_000, 5, flags=1)])
    reference = writer.finalize()

    assert reference.relative_path == "session-1/events.parquet"
    assert reference.event_count == 2
    assert reference.first_timestamp_ns == 1_000_000_000
    assert reference.last_timestamp_ns == 1_500_000_000
    table = pq.read_table(tmp_path / "raw_sessions" / reference.relative_path)
    assert table.column("timestamp_ns").to_pylist() == [1_000_000_000, 1_500_000_000]
    assert table.column("raw_dx").to_pylist() == [4, 5]
