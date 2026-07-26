from __future__ import annotations

import logging
import os
import sys
import time
from pathlib import Path

from mouselearn.diagnostics.checks import database_check, display_check, environment_checks, filesystem_check
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
