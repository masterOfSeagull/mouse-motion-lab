from __future__ import annotations

import logging
import os
import sys
import time
from pathlib import Path

from mouselearn.diagnostics.checks import database_check, display_check, environment_checks, filesystem_check
from mouselearn.preprocessing import PreprocessingSpec, preprocess_dataset_snapshot
from mouselearn.models.training import build_baseline_model
from mouselearn.models import ConditionalFlowConfig, train_conditional_flow
from mouselearn.domain.events import WorkerEvent
from mouselearn.storage.database import connect, migrate
from mouselearn.storage.paths import database_path, data_root, initialize_data_root
from mouselearn.storage.repositories import Repositories


def _emit(event: WorkerEvent) -> None:
    print(event.to_jsonl(), flush=True)


def _persist_then_emit(repos: Repositories, event: WorkerEvent) -> None:
    if event.event == "started": repos.update_job(event.job_id, status="running", progress=0, stage="starting")
    elif event.event == "stage_changed": repos.update_job(event.job_id, stage=event.stage)
    elif event.event == "progress": repos.update_job(event.job_id, progress=event.progress, stage=event.stage)
    elif event.event == "completed": repos.update_job(event.job_id, status="completed", progress=100, stage="complete")
    elif event.event in {"failed", "cancelled"}: repos.update_job(event.job_id, status=event.event, error=event.message)
    _emit(event)


def run_diagnostic(job_id: str, root: Path | None = None) -> int:
    root = initialize_data_root(root or data_root())
    conn = connect(database_path(root))
    logger = logging.getLogger("mousemotionlab.worker")
    try:
        migrate(conn)
        repos = Repositories(conn)
        _persist_then_emit(repos, WorkerEvent(event="started", job_id=job_id, message="Diagnostic started"))
        stages = [
            ("database", database_check(database_path(root))),
            ("storage", filesystem_check(root)),
            ("environment", None),
            ("display", display_check()),
        ]
        for index, (stage, result) in enumerate(stages, start=1):
            _persist_then_emit(repos, WorkerEvent(event="stage_changed", job_id=job_id, stage=stage))
            results = environment_checks() if stage == "environment" else [result]
            for item in results:
                _persist_then_emit(repos, WorkerEvent(event="metric", job_id=job_id, stage=stage, name=item.name, value=item.value))
                if item.warning:
                    _persist_then_emit(repos, WorkerEvent(event="warning", job_id=job_id, stage=stage, message=item.warning, name=item.name))
            _persist_then_emit(repos, WorkerEvent(event="progress", job_id=job_id, stage=stage, progress=index * 25))
            delay = int(os.environ.get("MOUSE_MOTION_LAB_DIAGNOSTIC_STAGE_DELAY_MS", "0"))
            if delay > 0:
                time.sleep(delay / 1000)
        _persist_then_emit(repos, WorkerEvent(event="completed", job_id=job_id, message="Diagnostic completed"))
        return 0
    except Exception as exc:
        logger.exception("diagnostic worker failed")
        try:
            _persist_then_emit(Repositories(conn), WorkerEvent(event="failed", job_id=job_id, message=str(exc)))
        except Exception:
            logger.exception("could not persist failure")
        return 1
    finally:
        conn.close()


def run_preprocessing(job_id: str, snapshot_id: str, root: Path | None = None) -> int:
    """Run representation construction in a worker process, never on the GUI thread."""
    root = initialize_data_root(root or data_root())
    conn = connect(database_path(root))
    logger = logging.getLogger("mousemotionlab.worker")
    try:
        migrate(conn)
        repos = Repositories(conn)
        _persist_then_emit(repos, WorkerEvent(event="started", job_id=job_id, message="Preprocessing started"))
        _persist_then_emit(repos, WorkerEvent(event="stage_changed", job_id=job_id, stage="verifying"))
        _persist_then_emit(repos, WorkerEvent(event="progress", job_id=job_id, stage="verifying", progress=10))
        _persist_then_emit(repos, WorkerEvent(event="stage_changed", job_id=job_id, stage="representing"))
        snapshot = repos.dataset_snapshot(snapshot_id)
        spec = PreprocessingSpec(**snapshot["preprocessing_config"])
        result = preprocess_dataset_snapshot(root, database_path(root), snapshot_id, spec)
        _persist_then_emit(repos, WorkerEvent(event="metric", job_id=job_id, stage="representing", name="processed_trials", value=result["processed_trial_count"]))
        _persist_then_emit(repos, WorkerEvent(event="metric", job_id=job_id, stage="representing", name="resampling_max_error", value=result["resampling"]["max_error"]))
        _persist_then_emit(repos, WorkerEvent(event="progress", job_id=job_id, stage="representing", progress=95))
        _persist_then_emit(repos, WorkerEvent(event="completed", job_id=job_id, message=f"Preprocessing complete: {result['processed_trial_count']} trials"))
        return 0
    except Exception as exc:
        logger.exception("preprocessing worker failed")
        try:
            _persist_then_emit(Repositories(conn), WorkerEvent(event="failed", job_id=job_id, message=str(exc)))
        except Exception:
            logger.exception("could not persist preprocessing failure")
        return 1
    finally:
        conn.close()


def run_baseline(job_id: str, preprocessing_run_id: str, model_type: str, root: Path | None = None) -> int:
    """Fit, validate, and publish a baseline outside the GUI process."""
    root = initialize_data_root(root or data_root())
    conn = connect(database_path(root))
    logger = logging.getLogger("mousemotionlab.worker")
    try:
        migrate(conn)
        repos = Repositories(conn)
        _persist_then_emit(repos, WorkerEvent(event="started", job_id=job_id, message=f"Building {model_type} baseline"))
        _persist_then_emit(repos, WorkerEvent(event="stage_changed", job_id=job_id, stage="loading_dataset"))
        _persist_then_emit(repos, WorkerEvent(event="progress", job_id=job_id, stage="loading_dataset", progress=10))
        _persist_then_emit(repos, WorkerEvent(event="stage_changed", job_id=job_id, stage="fitting"))
        result = build_baseline_model(root, database_path(root), preprocessing_run_id, model_type)
        report = result["validation"]
        _persist_then_emit(repos, WorkerEvent(event="metric", job_id=job_id, stage="validating", name="held_out_samples", value=report["held_out_sample_count"]))
        _persist_then_emit(repos, WorkerEvent(event="metric", job_id=job_id, stage="validating", name="endpoint_projection_rate", value=report["endpoint_projection_rate"]))
        _persist_then_emit(repos, WorkerEvent(event="progress", job_id=job_id, stage="publishing", progress=95))
        _persist_then_emit(repos, WorkerEvent(event="completed", job_id=job_id, message=f"{model_type} baseline ready: {result['model_id'][:8]}"))
        return 0
    except Exception as exc:
        logger.exception("baseline worker failed")
        try:
            _persist_then_emit(Repositories(conn), WorkerEvent(event="failed", job_id=job_id, message=str(exc)))
        except Exception:
            logger.exception("could not persist baseline failure")
        return 1
    finally:
        conn.close()


def run_flow_training(job_id: str, preprocessing_run_id: str, preset: str, root: Path | None = None) -> int:
    root = initialize_data_root(root or data_root())
    conn = connect(database_path(root))
    logger = logging.getLogger("mousemotionlab.worker")
    presets = {
        "small": ConditionalFlowConfig(hidden_size=96, hidden_layers=2, epochs=120, checkpoint_every=20),
        "standard": ConditionalFlowConfig(),
    }
    try:
        config = presets[preset]
        migrate(conn)
        repos = Repositories(conn)
        _persist_then_emit(repos, WorkerEvent(event="started", job_id=job_id, message=f"Conditional-flow {preset} training started"))
        _persist_then_emit(repos, WorkerEvent(event="stage_changed", job_id=job_id, stage="training"))

        def progress(epoch: int, metrics: dict[str, float]) -> None:
            percent = min(94, round(epoch / config.epochs * 94))
            if epoch == 1 or epoch % 5 == 0 or epoch == config.epochs:
                _persist_then_emit(repos, WorkerEvent(event="progress", job_id=job_id, stage="training", progress=percent))
                _persist_then_emit(repos, WorkerEvent(event="metric", job_id=job_id, stage="training", name="validation_loss", value=metrics["validation_loss"]))

        result = train_conditional_flow(root, database_path(root), preprocessing_run_id, config, progress, job_id)
        _persist_then_emit(repos, WorkerEvent(event="progress", job_id=job_id, stage="validating", progress=98))
        _persist_then_emit(repos, WorkerEvent(event="completed", job_id=job_id, message=f"Conditional flow ready: {result['model_id'][:8]}"))
        return 0
    except Exception as exc:
        logger.exception("conditional-flow worker failed")
        try:
            _persist_then_emit(Repositories(conn), WorkerEvent(event="failed", job_id=job_id, message=str(exc)))
        except Exception:
            logger.exception("could not persist flow-training failure")
        return 1
    finally:
        conn.close()
