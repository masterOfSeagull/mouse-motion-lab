from __future__ import annotations

from pathlib import Path

import pytest
from PySide6.QtCore import QObject, QUrl
from PySide6.QtCore import QTimer
from PySide6.QtQml import QQmlApplicationEngine

from mouselearn.storage.bootstrap import initialize
from mouselearn.storage.database import connect
from mouselearn.storage.repositories import Repositories
from mouselearn.ui.controllers import AppController, JobController


def test_qml_shell_loads(qapp, data_root) -> None:
    root, db, version = initialize(data_root)
    controller = AppController(root, db, version)
    engine = QQmlApplicationEngine()
    engine.rootContext().setContextProperty("appController", controller)
    qml = Path(__file__).parents[1] / "apps" / "control_panel" / "qml" / "Main.qml"
    engine.load(QUrl.fromLocalFile(str(qml)))
    assert engine.rootObjects()
    window = engine.rootObjects()[0]
    assert window.property("minimumWidth") == 720
    assert window.property("minimumHeight") == 450
    assert window.property("uiScale") == pytest.approx(980 / 1180)
    zoom = window.findChild(QObject, "trajectoryZoom")
    viewport = window.findChild(QObject, "trajectoryViewport")
    horizontal_scroll = window.findChild(QObject, "trajectoryHorizontalScroll")
    vertical_scroll = window.findChild(QObject, "trajectoryVerticalScroll")
    assert zoom is not None and zoom.property("from") == 1.0 and zoom.property("to") == 4.0
    assert viewport is not None
    assert horizontal_scroll is not None and vertical_scroll is not None


def test_qprocess_job_completes(qtbot, data_root) -> None:
    _, db, _ = initialize(data_root)
    controller = JobController(db)
    controller.startDiagnostic()
    qtbot.waitUntil(lambda: controller.process is not None and controller.process.state().value == 0, timeout=15000)
    conn = connect(db)
    try:
        assert Repositories(conn).job(controller.jobId)["status"] == "completed"
    finally:
        conn.close()


def test_qprocess_job_cancellation(qtbot, data_root, monkeypatch) -> None:
    monkeypatch.setenv("MOUSE_MOTION_LAB_DIAGNOSTIC_STAGE_DELAY_MS", "1000")
    _, db, _ = initialize(data_root)
    controller = JobController(db)
    controller.startDiagnostic()
    QTimer.singleShot(100, controller.cancel)
    qtbot.waitUntil(lambda: controller.process is not None and controller.process.state().value == 0, timeout=5000)
    conn = connect(db)
    try:
        assert Repositories(conn).job(controller.jobId)["status"] == "cancelled"
    finally:
        conn.close()
