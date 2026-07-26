"""Snapshot-bound baseline fitting, validation, and artifact publication."""
from __future__ import annotations

import hashlib
import json
import platform
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np

from mouselearn.datasets.snapshots import current_code_revision
from mouselearn.storage.database import connect, migrate
from mouselearn.storage.repositories import Repositories

from .dataset import load_processed_dataset
from .pca_mixture import PcaMixtureConfig, PcaMixtureGenerator
from .retrieval import RetrievalConfig, RetrievalGenerator
from .validation import validate_baseline
from .conditional_flow import ConditionalFlowGenerator


class BaselineTrainingError(RuntimeError):
    pass


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1_048_576), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")


def load_generator(source: Path):
    manifest = json.loads((source / "model.json").read_text(encoding="utf-8"))
    if manifest.get("model_type") == "retrieval":
        return RetrievalGenerator.load(source)
    if manifest.get("model_type") == "pca_mixture":
        return PcaMixtureGenerator.load(source)
    if manifest.get("model_type") == "conditional_flow":
        return ConditionalFlowGenerator.load(source)
    raise ValueError("unsupported model type in manifest")


def build_baseline_model(
    root: Path, database: Path, preprocessing_run_id: str, model_type: str,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Fit only the training split and publish only after held-out correctness passes."""
    dataset = load_processed_dataset(root, database, preprocessing_run_id)
    config = dict(config or {})
    if model_type == "retrieval":
        generator = RetrievalGenerator(RetrievalConfig(**config))
    elif model_type == "pca_mixture":
        generator = PcaMixtureGenerator(PcaMixtureConfig(**config))
    else:
        raise BaselineTrainingError(f"unsupported baseline model type: {model_type}")
    resolved_config = dict(generator.config.__dict__)
    revision = current_code_revision()
    conn = connect(database)
    model_id = ""
    model_dir: Path | None = None
    try:
        migrate(conn)
        repos = Repositories(conn)
        model_id = repos.create_baseline_model_draft(
            f"{model_type.replace('_', ' ').title()} {preprocessing_run_id[:8]}", model_type,
            dataset.snapshot_id, preprocessing_run_id, resolved_config, revision,
        )
        model_dir = root / "models" / model_id
        generator.fit(dataset)
        generator.save(model_dir)
        report = validate_baseline(generator, dataset)
        if not report["passed"]:
            raise BaselineTrainingError("baseline failed held-out correctness or determinism validation")
        report.update({"model_id": model_id, "code_revision": revision, "config": resolved_config})
        validation_path = model_dir / "validation.json"
        _json(validation_path, report)
        environment = {
            "python": sys.version.split()[0], "platform": platform.platform(), "numpy": np.__version__,
        }
        provenance_path = model_dir / "provenance.json"
        _json(provenance_path, {
            "schema_version": 1, "model_id": model_id, "model_type": model_type,
            "dataset_snapshot_id": dataset.snapshot_id, "preprocessing_run_id": preprocessing_run_id,
            "code_revision": revision, "config": resolved_config, "training_seed": 0, "environment": environment,
        })
        manifest_path = model_dir / "model.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest.update({
            "model_id": model_id, "dataset_snapshot_id": dataset.snapshot_id,
            "preprocessing_run_id": preprocessing_run_id, "code_revision": revision,
            "training_seed": 0, "environment": environment, "validation_sha256": _digest(validation_path),
        })
        _json(manifest_path, manifest)
        repos.finalize_baseline_model(
            model_id, manifest_path.relative_to(root).as_posix(), _digest(manifest_path),
            validation_path.relative_to(root).as_posix(), _digest(validation_path),
        )
        return {"model_id": model_id, "model_type": model_type, "model_path": model_dir.relative_to(root).as_posix(), "validation": report}
    except Exception as exc:
        if model_id:
            try:
                Repositories(conn).discard_baseline_model_draft(model_id, str(exc))
            except Exception:
                pass
        if model_dir is not None and model_dir.is_dir():
            shutil.rmtree(model_dir, ignore_errors=True)
        if isinstance(exc, BaselineTrainingError):
            raise
        raise BaselineTrainingError(str(exc)) from exc
    finally:
        conn.close()


__all__ = ["BaselineTrainingError", "build_baseline_model", "load_generator"]
