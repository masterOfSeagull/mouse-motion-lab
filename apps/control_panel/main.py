from __future__ import annotations

import os
import sys
from pathlib import Path

from PySide6.QtCore import QUrl
from PySide6.QtGui import QFont, QFontDatabase, QGuiApplication
from PySide6.QtQml import QQmlApplicationEngine

from mouselearn.storage.bootstrap import initialize
from mouselearn.storage.logging import configure_json_logging
from mouselearn.ui.controllers import AppController


def configure_application_font(app: QGuiApplication) -> None:
    """PySide wheels do not bundle fonts; use the Windows system UI face explicitly."""
    if sys.platform == "win32":
        for filename in ("segoeui.ttf", "segoeuib.ttf"):
            path = Path(r"C:\Windows\Fonts") / filename
            if path.is_file():
                QFontDatabase.addApplicationFont(str(path))
    app.setFont(QFont("Segoe UI", 10))


def main(argv: list[str] | None = None) -> int:
    app = QGuiApplication(argv or sys.argv)
    configure_application_font(app)
    root, database, schema_version = initialize()
    configure_json_logging(root / "logs" / "control-panel.jsonl")
    controller = AppController(root, database, schema_version)
    engine = QQmlApplicationEngine()
    engine.rootContext().setContextProperty("appController", controller)
    qml = Path(__file__).parent / "qml" / "Main.qml"
    engine.load(QUrl.fromLocalFile(str(qml)))
    if not engine.rootObjects():
        return 1
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
