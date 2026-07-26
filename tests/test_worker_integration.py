from __future__ import annotations

import os
import subprocess
import sys

from mouselearn.domain.events import parse_event
from mouselearn.storage.bootstrap import initialize
from mouselearn.storage.database import connect
from mouselearn.storage.repositories import Repositories


def test_worker_persists_lifecycle_and_emits_jsonl(data_root) -> None:
    _, db, _ = initialize(data_root)
    conn = connect(db)
    try:
        job_id = Repositories(conn).create_job("diagnostic")
    finally:
        conn.close()
    env = os.environ | {"MOUSE_MOTION_LAB_DATA_ROOT": str(data_root), "PYTHONUNBUFFERED": "1"}
    result = subprocess.run([sys.executable, "-m", "apps.worker", "diagnostic", "--job-id", job_id], text=True, capture_output=True, env=env, check=False)
    assert result.returncode == 0, result.stderr
    events = [parse_event(line) for line in result.stdout.splitlines()]
    assert events[0].event == "started"
    assert events[-1].event == "completed"
    conn = connect(db)
    try:
        assert Repositories(conn).job(job_id)["status"] == "completed"
    finally:
        conn.close()
