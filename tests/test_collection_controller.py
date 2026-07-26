from __future__ import annotations

from mouselearn.collection.controller import CollectionController
from mouselearn.collection.targets import ContinuousUniformTargetScheduler
from mouselearn.domain.collection import CollectionSessionPlan
from mouselearn.storage.bootstrap import initialize
from mouselearn.storage.database import connect
from mouselearn.storage.repositories import Repositories


class _Window:
    def devicePixelRatio(self) -> float:  # noqa: N802 - Qt API spelling
        return 1.0

    def winId(self) -> int:  # noqa: N802 - Qt API spelling
        return 1


class _Capture:
    def client_to_screen(self, _window_handle: int, x: int, y: int) -> tuple[int, int]:
        return x, y

    def cursor_position(self) -> tuple[int, int]:
        return 100, 100

    def qpc_now(self) -> int:
        return 1_000_000

    def stats(self):
        class _Stats:
            qpc_frequency_hz = 1_000_000

        return _Stats()


def test_protocol_three_target_with_no_requested_distance_creates_a_trial(qtbot, tmp_path) -> None:
    """Uniform Protocol-3 conditions intentionally leave requested distance unset."""
    root, database, _ = initialize(tmp_path)
    conn = connect(database)
    try:
        repositories = Repositories(conn)
        session_id = repositories.create_collection_session(CollectionSessionPlan(
            display_name="protocol-three controller test", planned_trials=1, random_seed=3,
        ))
        repositories.transition_collection_session(session_id, "active")
    finally:
        conn.close()

    controller = CollectionController(root, database)
    controller._state = "active"
    controller._session_id = session_id
    controller._window = _Window()
    controller._capture = _Capture()
    controller._canvas_width = 1280
    controller._canvas_height = 720
    controller._scheduler = ContinuousUniformTargetScheduler(3)

    controller._begin_trial()

    assert controller.targetVisible
    assert controller._trial_id
    conn = connect(database)
    try:
        assert conn.execute("SELECT count(*) FROM trials WHERE session_id=?", (session_id,)).fetchone()[0] == 1
    finally:
        conn.close()
