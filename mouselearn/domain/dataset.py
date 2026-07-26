"""Strict contracts for immutable, session-held-out dataset snapshots."""
from __future__ import annotations

import hashlib
from typing import Any
from uuid import UUID

from pydantic import Field, model_validator

from .config import StrictModel


class SessionHeldOutSplit(StrictModel):
    """Deterministic split configuration; trials from a session never cross splits."""

    seed: int = Field(default=0, ge=0)
    train_fraction: float = Field(default=0.70, gt=0, lt=1)
    validation_fraction: float = Field(default=0.15, ge=0, lt=1)
    test_fraction: float = Field(default=0.15, ge=0, lt=1)

    @model_validator(mode="after")
    def fractions_sum_to_one(self) -> "SessionHeldOutSplit":
        if abs(self.train_fraction + self.validation_fraction + self.test_fraction - 1.0) > 1e-9:
            raise ValueError("split fractions must sum to one")
        return self


class DatasetSnapshotPlan(StrictModel):
    name: str = Field(min_length=1, max_length=160)
    description: str = Field(default="", max_length=2_000)
    session_ids: tuple[UUID, ...] = ()
    split: SessionHeldOutSplit = Field(default_factory=SessionHeldOutSplit)
    preprocessing_config: dict[str, Any] = Field(default_factory=dict)
    feature_schema_version: int = Field(default=1, ge=1)

    @model_validator(mode="after")
    def sessions_are_unique(self) -> "DatasetSnapshotPlan":
        if len(set(self.session_ids)) != len(self.session_ids):
            raise ValueError("session_ids must be unique")
        return self


def session_held_out_assignments(session_ids: list[str], seed: int) -> tuple[dict[str, str], list[str]]:
    """Deterministically keep every trial from a session in exactly one split."""
    ordered = sorted(session_ids, key=lambda item: hashlib.sha256(f"{seed}:{item}".encode()).hexdigest())
    count = len(ordered)
    if count == 0:
        raise ValueError("no eligible completed sessions were selected")
    warnings: list[str] = []
    if count < 3:
        warnings.append(
            f"Only {count} eligible session{' is' if count == 1 else 's are'} available; "
            "validation/test splits are not independent."
        )
    validation_count = 0 if count < 3 else max(1, int(count * 0.15 + 0.5))
    test_count = 0 if count == 1 else max(1, int(count * 0.15 + 0.5))
    while validation_count + test_count >= count:
        if validation_count > 0:
            validation_count -= 1
        elif test_count > 0:
            test_count -= 1
        else:
            break
    assignments: dict[str, str] = {}
    for index, session_id in enumerate(ordered):
        assignments[session_id] = "test" if index < test_count else "validation" if index < test_count + validation_count else "train"
    return assignments, warnings


__all__ = ["DatasetSnapshotPlan", "SessionHeldOutSplit", "session_held_out_assignments"]
