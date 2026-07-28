"""Validated model comparison and promotion controller."""
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QObject, Property, Signal, Slot

from mouselearn.models import PromotionError, compare_models, promote_model
from mouselearn.export import export_conditional_flow, export_pca_mixture
from mouselearn.storage.database import connect
from mouselearn.storage.repositories import Repositories


class RegistryController(QObject):
    modelsChanged = Signal()
    messageChanged = Signal()

    def __init__(self, root: Path, database: Path, parent: QObject | None = None):
        super().__init__(parent)
        self.root, self.database = root, database
        self._models: list[dict] = []
        self._message = "Promotion requires an intact passing validation report."
        self.refresh()

    @Property("QVariantList", notify=modelsChanged)
    def models(self) -> list[dict]:
        return self._models

    @Property(str, notify=messageChanged)
    def message(self) -> str:
        return self._message

    @Slot()
    def refresh(self) -> None:
        self._models = compare_models(self.root, self.database)
        self.modelsChanged.emit()

    @Slot(str)
    def promote(self, model_id: str) -> None:
        try:
            promote_model(self.root, self.database, model_id)
            self._message = f"Model {model_id[:8]} is now active."
        except PromotionError as exc:
            self._message = f"Promotion blocked: {exc}"
        self.messageChanged.emit()
        self.refresh()

    @Slot(str)
    def exportModel(self, model_id: str) -> None:
        try:
            conn = connect(self.database)
            try:
                repos = Repositories(conn)
                model = repos.baseline_model(model_id)
                if model["model_type"] not in {"conditional_flow", "pca_mixture"} or model["status"] != "ready":
                    raise ValueError("only ready conditional-flow and PCA-mixture models support portable export")
                destination = self.root / "exports" / f"{model_id}-portable"
                exporter = export_pca_mixture if model["model_type"] == "pca_mixture" else export_conditional_flow
                exporter((self.root / model["manifest_relative_path"]).parent, destination)
                repos.audit("model", model_id, "exported", {"destination": str(destination)})
            finally:
                conn.close()
            self._message = f"Portable package exported to {destination}"
        except (FileExistsError, KeyError, OSError, RuntimeError, ValueError) as exc:
            self._message = f"Export blocked: {exc}"
        self.messageChanged.emit()


__all__ = ["RegistryController"]
