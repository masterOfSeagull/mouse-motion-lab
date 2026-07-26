from __future__ import annotations

from pathlib import Path

from .database import connect, migrate
from .paths import database_path, initialize_data_root
from .repositories import Repositories


def initialize(root: Path | None = None) -> tuple[Path, Path, int]:
    root = initialize_data_root(root)
    db_path = database_path(root)
    conn = connect(db_path)
    try:
        version = migrate(conn)
        Repositories(conn).reconcile_interrupted_jobs()
    finally:
        conn.close()
    return root, db_path, version
