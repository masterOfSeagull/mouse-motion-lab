from __future__ import annotations

from mouselearn.runtime import PlaybackController, generated_playback_active


def _trajectory(duration_ns: int = 20_000_000) -> dict:
    return {"points": [
        {"x": 10.0, "y": 20.0, "time_ns": 0},
        {"x": 15.0, "y": 25.0, "time_ns": duration_ns // 2},
        {"x": 30.0, "y": 40.0, "time_ns": duration_ns},
    ]}


def test_playback_never_starts_unarmed_and_auto_disarms(qapp, qtbot) -> None:
    controller = PlaybackController()
    controller.setTrajectory(_trajectory())
    controller.start()
    assert controller.state == "disarmed"
    assert not generated_playback_active()
    controller.arm(); assert controller.state == "armed"
    controller.start(); assert controller.state == "playing" and generated_playback_active()
    qtbot.waitUntil(lambda: controller.state == "disarmed", timeout=1000)
    assert (controller.x, controller.y, controller.progress) == (30.0, 40.0, 1.0)
    assert not generated_playback_active()


def test_substantial_physical_movement_aborts_and_retains_no_arming(qapp, qtbot) -> None:
    controller = PlaybackController()
    controller.setTrajectory(_trajectory(1_000_000_000))
    controller.arm(); controller.start()
    controller.notifyPhysicalDelta(20, 0)
    assert controller.state == "disarmed"
    assert "physical mouse" in controller.message
    qtbot.waitUntil(lambda: not generated_playback_active(), timeout=500)
    controller.shutdown()


def test_escape_abort_path_clears_generated_marker(qapp, qtbot) -> None:
    controller = PlaybackController()
    controller.setTrajectory(_trajectory(1_000_000_000))
    controller.arm(); controller.start(); controller.abortFromUi()
    assert controller.state == "disarmed"
    assert "Emergency stop" in controller.message
    assert not generated_playback_active()
    controller.shutdown()
