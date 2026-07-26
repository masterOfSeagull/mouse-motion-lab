from __future__ import annotations

import importlib.util
import platform
import sqlite3
import sys
import ctypes
from dataclasses import asdict, dataclass
from pathlib import Path

from mouselearn.collection.native import native_library_path
from mouselearn.storage.database import connect, migrate


@dataclass(frozen=True)
class CheckResult:
    name: str
    ok: bool
    value: object
    warning: str | None = None


def filesystem_check(root: Path) -> CheckResult:
    root.mkdir(parents=True, exist_ok=True)
    probe = root / ".write_probe"
    try:
        probe.write_text("ok", encoding="ascii")
        probe.unlink()
        return CheckResult("filesystem", True, str(root))
    except OSError as exc:
        return CheckResult("filesystem", False, str(root), str(exc))


def database_check(path: Path) -> CheckResult:
    try:
        conn = connect(path)
        try:
            version = migrate(conn)
            integrity = conn.execute("PRAGMA integrity_check").fetchone()[0]
            return CheckResult("database", integrity == "ok", {"schema_version": version, "integrity": integrity}, None if integrity == "ok" else integrity)
        finally:
            conn.close()
    except sqlite3.Error as exc:
        return CheckResult("database", False, str(path), str(exc))


def environment_checks() -> list[CheckResult]:
    packages = {name: importlib.util.find_spec(name) is not None for name in ("PySide6", "pyarrow", "torch", "onnxruntime")}
    cuda = False
    if packages["torch"]:
        import torch  # optional dependency, only imported when installed
        cuda = bool(torch.cuda.is_available())
    return [
        CheckResult("python", sys.version.split()[0].startswith("3.12"), sys.version.split()[0], "Python 3.12 is required" if not sys.version.split()[0].startswith("3.12") else None),
        CheckResult("pyside6", packages["PySide6"], packages["PySide6"], "PySide6 unavailable" if not packages["PySide6"] else None),
        CheckResult("pyarrow", packages["pyarrow"], packages["pyarrow"], "required for collection persistence" if not packages["pyarrow"] else None),
        CheckResult("pytorch", packages["torch"], packages["torch"], "optional dependency unavailable" if not packages["torch"] else None),
        CheckResult("cuda", cuda, cuda, "CUDA unavailable (optional)" if not cuda else None),
        CheckResult("onnxruntime", packages["onnxruntime"], packages["onnxruntime"], "optional dependency unavailable" if not packages["onnxruntime"] else None),
        CheckResult("platform", True, {"system": platform.system(), "release": platform.release()}),
        CheckResult("native_raw_input", native_library_path() is not None, str(native_library_path() or ""), "run tools/build-native.ps1 before collection" if native_library_path() is None else None),
    ]


def display_check() -> CheckResult:
    # Qt is deliberately not initialized in the worker: this remains safe in headless CI.
    if sys.platform == "win32":
        user32 = ctypes.windll.user32
        monitors = user32.GetSystemMetrics(80)  # SM_CMONITORS
        dpi = user32.GetDpiForSystem() if hasattr(user32, "GetDpiForSystem") else 96
        return CheckResult("display", monitors > 0, {"monitor_count": monitors, "system_dpi": dpi})
    return CheckResult("display", True, {"monitor_count": "not queried", "system_dpi": "not queried"})
