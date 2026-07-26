from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

from mouselearn.collection.controller import CollectionController
from mouselearn.collection.native import MML_CAPTURE_OK


class _Capture:
    def drain(self):
        return MML_CAPTURE_OK, []

    def stats(self):
        return SimpleNamespace(
            observed_events=0,
            buffered_events=0,
            overflow_events=0,
            qpc_frequency_hz=1_000_000,
        )


class _Writer:
    failure = None


def test_capture_timer_activates_target_when_qt_presentation_callback_is_missing(qtbot, tmp_path) -> None:
    """A visible target must not remain unclickable when frameSwapped is omitted."""
    controller = CollectionController(tmp_path, tmp_path / "mouselearn.sqlite3")
    controller._state = "active"
    controller._capture = _Capture()
    controller._writer = _Writer()
    controller._pending_target = object()
    controller._target_activation_deadline_ns = 0
    controller._activate_presented_target = Mock()  # type: ignore[method-assign]

    controller._drain_capture()

    controller._activate_presented_target.assert_called_once_with("presentation_fallback")
