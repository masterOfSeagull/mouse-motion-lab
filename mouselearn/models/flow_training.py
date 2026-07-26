"""Tracked conditional-flow training and publication."""
from __future__ import annotations

import hashlib
import json
import platform
import shutil
import sys
from pathlib import Path
from typing import Any, Callable

import numpy as np

from mouselearn.datasets.snapshots import current_code_revision
from mouselearn.storage.database import connect, migrate
from mouselearn.storage.repositories import Repositories

from .conditional_flow import ConditionalFlowConfig, ConditionalFlowGenerator
from .dataset import load_processed_dataset
from .training import BaselineTrainingError
from .validation import validate_baseline


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")


def train_conditional_flow(
    root: Path, database: Path, preprocessing_run_id: str, config: ConditionalFlowConfig = ConditionalFlowConfig(),
    progress: Callable[[int, dict[str, float]], None] | None = None, job_id: str | None = None,
) -> dict[str, Any]:
    dataset = load_processed_dataset(root, database, preprocessing_run_id)
    revision = current_code_revision()
    resolved = dict(config.__dict__)
    conn = connect(database)
    experiment_id = model_id = ""
    model_dir: Path | None = None
    try:
        migrate(conn)
        repos = Repositories(conn)
        experiment_id = repos.create_experiment(
            f"Conditional flow {preprocessing_run_id[:8]}", dataset.snapshot_id, preprocessing_run_id, resolved, config.seed, job_id,
        )
        repos.start_experiment(experiment_id)
        model_id = repos.create_flow_model_draft(
            f"Conditional Flow {preprocessing_run_id[:8]}", dataset.snapshot_id, preprocessing_run_id, resolved, revision,
        )
        checkpoint_dir = root / "experiments" / experiment_id / "checkpoints"

        def on_epoch(epoch: int, metrics: dict[str, float]) -> None:
            checkpoint = checkpoint_dir / f"epoch-{epoch:04d}.pt"
            relative = checkpoint.relative_to(root).as_posix() if checkpoint.is_file() else None
            repos.update_experiment_metrics(experiment_id, epoch, metrics, relative)
            if progress:
                progress(epoch, metrics)

        generator = ConditionalFlowGenerator(config).fit(dataset, on_epoch, checkpoint_dir)
        final_checkpoint = checkpoint_dir / f"epoch-{config.epochs:04d}.pt"
        if final_checkpoint.is_file() and generator.history:
            repos.update_experiment_metrics(
                experiment_id, config.epochs, generator.history[-1], final_checkpoint.relative_to(root).as_posix(),
            )
        model_dir = root / "models" / model_id
        generator.save(model_dir)
        report = validate_baseline(generator, dataset)
        if not report["passed"]:
            raise BaselineTrainingError("conditional flow failed held-out correctness or determinism validation")
        report.update({"model_id": model_id, "experiment_id": experiment_id, "code_revision": revision, "config": resolved})
        validation_path = model_dir / "validation.json"
        _write_json(validation_path, report)
        manifest_path = model_dir / "model.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest.update({
            "model_id": model_id, "experiment_id": experiment_id, "dataset_snapshot_id": dataset.snapshot_id,
            "preprocessing_run_id": preprocessing_run_id, "code_revision": revision, "training_seed": config.seed,
            "environment": {"python": sys.version.split()[0], "platform": platform.platform(), "numpy": np.__version__},
            "validation_sha256": _digest(validation_path),
        })
        _write_json(manifest_path, manifest)
        repos.finalize_baseline_model(
            model_id, manifest_path.relative_to(root).as_posix(), _digest(manifest_path),
            validation_path.relative_to(root).as_posix(), _digest(validation_path),
        )
        repos.complete_experiment(experiment_id, model_id)
        return {"experiment_id": experiment_id, "model_id": model_id, "validation": report, "history": generator.history}
    except Exception as exc:
        if experiment_id:
            try:
                Repositories(conn).fail_experiment(experiment_id, str(exc))
            except Exception:
                pass
        if model_id:
            try:
                Repositories(conn).discard_baseline_model_draft(model_id, str(exc))
            except Exception:
                pass
        if model_dir is not None and model_dir.is_dir():
            shutil.rmtree(model_dir, ignore_errors=True)
        raise
    finally:
        conn.close()


__all__ = ["train_conditional_flow"]
