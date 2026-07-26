from __future__ import annotations

import argparse
import logging
import sys

from mouselearn.jobs.worker import run_diagnostic, run_preprocessing
from mouselearn.storage.logging import configure_json_logging
from mouselearn.storage.paths import data_root, initialize_data_root


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="mousemotionlab-worker")
    command = parser.add_subparsers(dest="command", required=True)
    diagnostic = command.add_parser("diagnostic")
    diagnostic.add_argument("--job-id", required=True)
    preprocess = command.add_parser("preprocess")
    preprocess.add_argument("--job-id", required=True)
    preprocess.add_argument("--snapshot-id", required=True)
    args = parser.parse_args(argv)
    root = initialize_data_root(data_root())
    configure_json_logging(root / "logs" / "worker.jsonl", "mousemotionlab.worker")
    if args.command == "diagnostic":
        return run_diagnostic(args.job_id, root)
    if args.command == "preprocess":
        return run_preprocessing(args.job_id, args.snapshot_id, root)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
