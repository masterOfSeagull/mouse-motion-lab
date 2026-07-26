"""Model registry and local, data-only trajectory preview controller."""
from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
from PySide6.QtCore import QObject, Property, Signal, Slot
from PySide6.QtGui import QGuiApplication

from mouselearn.models import GenerationRequest, condition_vector, decode_output, load_generator
from mouselearn.storage.database import connect, migrate
from mouselearn.storage.repositories import Repositories


class GeneratorController(QObject):
    modelsChanged = Signal()
    trajectoryChanged = Signal()
    messageChanged = Signal()

    def __init__(
        self, root: Path, database: Path, start_baseline: Callable[[str, str], None], parent: QObject | None = None,
    ):
        super().__init__(parent)
        self.root, self.database, self._start_baseline = root, database, start_baseline
        self._models: list[dict] = []
        self._trajectory: dict = {}
        self._message = "Build a baseline from a completed representation, then generate a local preview."
        self.refresh()

    @Property("QVariantList", notify=modelsChanged)
    def models(self) -> list[dict]:
        return self._models

    @Property("QVariantMap", notify=trajectoryChanged)
    def trajectory(self) -> dict:
        return self._trajectory

    @Property(str, notify=messageChanged)
    def message(self) -> str:
        return self._message

    def _set_message(self, message: str) -> None:
        self._message = message
        self.messageChanged.emit()

    @Slot()
    def refresh(self) -> None:
        conn = connect(self.database)
        try:
            migrate(conn)
            self._models = Repositories(conn).baseline_models()
            for model in self._models:
                validation = model.get("validation_relative_path")
                if validation:
                    path = (self.root / validation).resolve()
                    try:
                        if path.is_relative_to(self.root.resolve()):
                            report = json.loads(path.read_text(encoding="utf-8"))
                            model["validation_summary"] = (
                                f"{report['held_out_sample_count']} held out · "
                                f"{report['endpoint_projection_rate'] * 100:.1f}% runtime endpoint corrections"
                            )
                    except (OSError, KeyError, ValueError, json.JSONDecodeError):
                        model["validation_summary"] = "Validation report is unreadable"
        finally:
            conn.close()
        self.modelsChanged.emit()

    @Slot(str, str)
    def buildBaseline(self, preprocessing_run_id: str, model_type: str) -> None:
        if not preprocessing_run_id:
            self._set_message("Select a completed preprocessing run first.")
            return
        if model_type not in {"retrieval", "pca_mixture"}:
            self._set_message("Choose a supported baseline type.")
            return
        self._start_baseline(preprocessing_run_id, model_type)
        self._set_message(f"Building {model_type.replace('_', ' ')} in a worker process.")

    @staticmethod
    def _virtual_desktop_bounds() -> tuple[float, float, float, float]:
        screens = QGuiApplication.screens()
        if not screens:
            return 0.0, 0.0, 1920.0, 1080.0
        bounds = screens[0].geometry()
        for screen in screens[1:]:
            bounds = bounds.united(screen.geometry())
        return float(bounds.x()), float(bounds.y()), float(bounds.width()), float(bounds.height())

    @Slot(str, float, float, float, float, float, int)
    def generate(
        self, model_id: str, start_x: float, start_y: float, target_x: float, target_y: float,
        radius: float, seed: int,
    ) -> None:
        try:
            conn = connect(self.database)
            try:
                model = Repositories(conn).baseline_model(model_id)
            finally:
                conn.close()
            if model["status"] != "ready" or not model["manifest_relative_path"] or not model["manifest_sha256"]:
                raise ValueError("selected model is not ready")
            manifest_path = (self.root / model["manifest_relative_path"]).resolve()
            if not manifest_path.is_relative_to(self.root.resolve()) or not manifest_path.is_file():
                raise ValueError("model artifact is missing")
            if hashlib.sha256(manifest_path.read_bytes()).hexdigest() != model["manifest_sha256"]:
                raise ValueError("model manifest hash changed")
            desktop_left, desktop_top, desktop_width, desktop_height = self._virtual_desktop_bounds()
            request = GenerationRequest(
                start_x, start_y, target_x, target_y, radius,
                desktop_left, desktop_top, desktop_width, desktop_height, random_seed=seed,
            )
            generator = load_generator(manifest_path.parent)
            thresholds = {"in_distribution_max": 3.0, "sparse_max": 6.0}
            validation_relative = model.get("validation_relative_path")
            if validation_relative:
                validation_path = (self.root / validation_relative).resolve()
                if validation_path.is_relative_to(self.root.resolve()) and validation_path.is_file():
                    thresholds.update(json.loads(validation_path.read_text(encoding="utf-8")).get("condition_distance_percentiles", {}))
            started = time.perf_counter_ns()
            batch = generator.generate_batch(condition_vector(request)[None, :], np.asarray([seed], dtype=np.uint64))
            score = float(batch.nearest_distances[0])
            condition_class = (
                "out_of_distribution" if score > float(thresholds["sparse_max"])
                else "sparse" if score > float(thresholds["in_distribution_max"])
                else "in_distribution"
            )
            result = decode_output(
                batch.outputs[0], request, condition_distance=float(batch.nearest_distances[0]),
                out_of_distribution=bool(batch.out_of_distribution[0]) or condition_class == "out_of_distribution",
            )
            elapsed_ms = (time.perf_counter_ns() - started) / 1_000_000
            self._trajectory = {
                "points": [{"x": sample.x, "y": sample.y, "time_ns": sample.relative_time_ns} for sample in result.samples],
                "start": {"x": request.start_x, "y": request.start_y},
                "target": {"x": request.target_center_x, "y": request.target_center_y, "radius": request.target_radius},
                "desktop": {
                    "left": request.virtual_desktop_left, "top": request.virtual_desktop_top,
                    "width": request.virtual_desktop_width, "height": request.virtual_desktop_height,
                },
                "duration_ms": result.movement_duration_ns / 1_000_000,
                "path_length": result.path_length, "peak_speed": result.peak_speed,
                "seed": seed, "inference_ms": elapsed_ms, "out_of_distribution": result.out_of_distribution,
                "condition_distance": result.condition_distance_score, "endpoint_projected": result.endpoint_projected,
                "condition_class": condition_class,
                "desktop_clipped_point_count": result.desktop_clipped_point_count,
            }
            self.trajectoryChanged.emit()
            warning = "" if condition_class == "in_distribution" else f" · {condition_class.replace('_', ' ')} request"
            self._set_message(f"Generated 64 points in {elapsed_ms:.2f} ms{warning}.")
        except (KeyError, OSError, RuntimeError, TypeError, ValueError) as exc:
            self._set_message(f"Could not generate preview: {exc}")


__all__ = ["GeneratorController"]
