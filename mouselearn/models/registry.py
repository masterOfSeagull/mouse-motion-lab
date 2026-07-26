"""Hash-verified promotion gates and model comparison reports."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from mouselearn.storage.database import connect, migrate
from mouselearn.storage.repositories import Repositories


class PromotionError(RuntimeError):
    pass


def _verified_report(root: Path, model: dict[str, Any]) -> dict[str, Any]:
    relative, expected = model.get("validation_relative_path"), model.get("validation_sha256")
    if not relative or not expected:
        raise PromotionError("model has no completed validation report")
    root = root.resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise PromotionError("validation report is missing")
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise PromotionError("validation report hash changed")
    report = json.loads(path.read_text(encoding="utf-8"))
    if not report.get("passed") or report.get("model_id") != model["id"]:
        raise PromotionError("validation report did not pass or belongs to another model")
    return report


def promote_model(root: Path, database: Path, model_id: str) -> None:
    conn = connect(database)
    try:
        migrate(conn)
        repos = Repositories(conn)
        model = repos.baseline_model(model_id)
        _verified_report(root, model)
        repos.promote_validated_model(model_id, model["validation_sha256"])
    except (KeyError, OSError, ValueError, json.JSONDecodeError) as exc:
        if isinstance(exc, PromotionError):
            raise
        raise PromotionError(str(exc)) from exc
    finally:
        conn.close()


def compare_models(root: Path, database: Path) -> list[dict[str, Any]]:
    conn = connect(database)
    try:
        migrate(conn)
        result = []
        for model in Repositories(conn).registry_models():
            try:
                report = _verified_report(root, model)
                result.append({
                    **model, "held_out_sample_count": report["held_out_sample_count"],
                    "duration_wasserstein_ns": report["movement_duration"]["wasserstein_ns"],
                    "path_wasserstein": report["path_length_ratio"]["wasserstein"],
                    "endpoint_projection_rate": report["endpoint_projection_rate"],
                    "ood_count": report["out_of_distribution_count"],
                })
            except PromotionError:
                result.append({**model, "validation_error": "Validation report unavailable or invalid"})
        return result
    finally:
        conn.close()


__all__ = ["PromotionError", "compare_models", "promote_model"]
