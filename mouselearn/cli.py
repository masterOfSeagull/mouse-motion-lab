from __future__ import annotations

import argparse

from mouselearn.diagnostics.checks import database_check, environment_checks, filesystem_check
from mouselearn.storage.bootstrap import initialize


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="mouselearn")
    parser.add_argument("command", choices=["doctor"])
    args = parser.parse_args(argv)
    root, database, version = initialize()
    checks = [filesystem_check(root), database_check(database), *environment_checks()]
    for result in checks:
        print(f"{'OK' if result.ok else 'WARN'} {result.name}: {result.value}" + (f" ({result.warning})" if result.warning else ""))
    print(f"schema_version={version}")
    return 0
