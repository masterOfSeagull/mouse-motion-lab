"""Build deterministic, immutable manifests without preprocessing raw trajectory data."""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

from mouselearn.domain.dataset import DatasetSnapshotPlan, session_held_out_assignments
from mouselearn.storage.database import connect, migrate
from mouselearn.storage.repositories import Repositories


class DatasetBuildError(RuntimeError):
    pass


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def current_code_revision() -> str:
    """Return the checkout revision when available; installed builds remain explicit."""
    try:
        root = Path(__file__).resolve().parents[2]
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, check=True, capture_output=True, text=True, timeout=2,
        )
        return result.stdout.strip() or "unknown"
    except (FileNotFoundError, subprocess.SubprocessError):
        return "unknown"


def build_dataset_snapshot(root: Path, database: Path, plan: DatasetSnapshotPlan) -> dict[str, Any]:
    """Create a manifest and make its database snapshot ready only after it is durable."""
    conn = connect(database)
    try:
        migrate(conn)
        repos = Repositories(conn)
        draft = repos.create_dataset_snapshot_draft(plan, current_code_revision())
        raw_root = (root / "raw_sessions").resolve()
        for raw_file in draft["raw_files"]:
            candidate = (raw_root / raw_file["relative_path"]).resolve()
            if not candidate.is_relative_to(raw_root) or not candidate.is_file():
                raise DatasetBuildError(f"raw evidence is missing: {raw_file['relative_path']}")
            actual = hashlib.sha256(candidate.read_bytes()).hexdigest()
            if actual != raw_file["sha256"]:
                raise DatasetBuildError(f"raw evidence hash changed: {raw_file['relative_path']}")
        manifest = draft["manifest"]
        manifest_path = root / "datasets" / draft["id"] / "manifest.json"
        manifest_path.parent.mkdir(parents=True, exist_ok=False)
        temporary_path = manifest_path.with_suffix(".json.tmp")
        try:
            temporary_path.write_text(_canonical_json(manifest) + "\n", encoding="utf-8")
            temporary_path.replace(manifest_path)
        except Exception:
            temporary_path.unlink(missing_ok=True)
            raise
        repos.finalize_dataset_snapshot(draft["id"], manifest_path.relative_to(root).as_posix(), _digest(manifest))
        return repos.dataset_snapshot(draft["id"])
    except Exception as exc:
        if isinstance(exc, DatasetBuildError):
            raise
        raise DatasetBuildError(str(exc)) from exc
    finally:
        conn.close()
