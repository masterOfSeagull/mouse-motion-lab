"""Validated line-oriented worker protocol."""
from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class WorkerEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")
    event: Literal["started", "stage_changed", "progress", "metric", "warning", "completed", "failed", "cancelled"]
    job_id: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    stage: str | None = None
    progress: int | None = Field(default=None, ge=0, le=100)
    message: str | None = None
    name: str | None = None
    value: Any | None = None

    def to_jsonl(self) -> str:
        return self.model_dump_json()


def parse_event(line: str) -> WorkerEvent:
    return WorkerEvent.model_validate(json.loads(line))
