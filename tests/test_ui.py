from __future__ import annotations

from pathlib import Path

import pytest
from PySide6.QtCore import QObject, QPointF, QUrl
from PySide6.QtCore import QTimer
from PySide6.QtQml import QQmlApplicationEngine
from PySide6.QtQuick import QQuickItem

from mouselearn.storage.bootstrap import initialize
from mouselearn.storage.database import connect
from mouselearn.storage.repositories import Repositories
from mouselearn.ui.controllers import AppController, JobController
from mouselearn.ui.generator import resolve_generation_seed


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
    generation_seed = window.findChild(QObject, "generationSeed")
    generator_page = window.findChild(QQuickItem, "generatorPage")
    experiments_page = window.findChild(QQuickItem, "experimentsPage")
    generator_data_choice = window.findChild(QQuickItem, "generatorTrainingDataChoice")
    generator_model_choice = window.findChild(QQuickItem, "generatorModelChoice")
    flow_data_choice = window.findChild(QQuickItem, "flowTrainingDataChoice")
    use_trained_model = window.findChild(QQuickItem, "useTrainedModelButton")
    assert zoom is not None and zoom.property("from") == 1.0 and zoom.property("to") == 4.0
    assert viewport is not None
    assert horizontal_scroll is not None and vertical_scroll is not None
    assert generation_seed is not None and generation_seed.property("from") == -1
    assert generator_page is not None and experiments_page is not None
    assert generator_data_choice is not None and generator_data_choice.property("hoverEnabled") is True
    assert generator_model_choice is not None and generator_model_choice.property("hoverEnabled") is True
    assert flow_data_choice is not None and flow_data_choice.property("hoverEnabled") is True
    assert use_trained_model is not None

    window.setProperty("width", 720)
    window.setProperty("height", 450)
    qapp.processEvents()
    for page_index, page, names in (
        (6, generator_page, ("generatorTrainingDataChoice", "buildPcaButton", "generatorModelChoice", "useTrainedModelButton", "generateButton", "fitTrajectoryButton", "trajectoryViewport")),
        (7, experiments_page, ("flowTrainingDataChoice", "trainStandardButton")),
    ):
        window.setProperty("currentPage", page_index)
        qapp.processEvents()
        for name in names:
            item = window.findChild(QQuickItem, name)
            assert item is not None
            origin = item.mapToItem(page, QPointF(0, 0))
            assert origin.x() >= -1
            assert origin.x() + item.width() <= page.width() + 1
            assert origin.y() >= -1
            assert origin.y() + item.height() <= page.height() + 1


def test_random_generation_seed_sentinel(monkeypatch) -> None:
    monkeypatch.setattr("mouselearn.ui.generator.secrets.randbelow", lambda upper: 123456789 if upper == 1_000_000_000 else -1)
    assert resolve_generation_seed(-1) == 123456789
    assert resolve_generation_seed(42) == 42
    with pytest.raises(ValueError, match="seed must be -1"):
        resolve_generation_seed(-2)


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
