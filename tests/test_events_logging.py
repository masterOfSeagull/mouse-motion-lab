from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from mouselearn.domain.events import WorkerEvent, parse_event
from mouselearn.storage.logging import configure_json_logging


def test_event_jsonl_validation() -> None:
    event = WorkerEvent(event="progress", job_id="abc", progress=50)
    assert parse_event(event.to_jsonl()).progress == 50
    with pytest.raises(ValidationError):
        parse_event('{"event":"progress","job_id":"abc","unknown":1}')


def test_json_logging(tmp_path) -> None:
    path = tmp_path / "logs" / "app.jsonl"
    logger = configure_json_logging(path, "tests.json")
    logger.info("hello")
    payload = json.loads(path.read_text(encoding="utf-8").strip())
    assert payload["message"] == "hello" and "+00:00" in payload["timestamp"]
