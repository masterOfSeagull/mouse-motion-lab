from __future__ import annotations

import argparse
import json
from pathlib import Path

from mouselearn.diagnostics.checks import database_check, environment_checks, filesystem_check
from mouselearn.export import OnnxFlowRuntime, export_conditional_flow
from mouselearn.models import GenerationRequest
from mouselearn.storage.bootstrap import initialize
from mouselearn.storage.database import connect
from mouselearn.storage.repositories import Repositories


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="mouselearn")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("doctor")
    export = commands.add_parser("export")
    export.add_argument("--model-id")
    export.add_argument("--destination", type=Path)
    generate = commands.add_parser("generate")
    generate.add_argument("--model", type=Path, required=True)
    generate.add_argument("--start", required=True)
    generate.add_argument("--target", required=True)
    generate.add_argument("--radius", type=float, required=True)
    generate.add_argument("--seed", type=int, default=42)
    generate.add_argument("--desktop", default="0,0,1920,1080")
    generate.add_argument("--click", action="store_true")
    generate.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    root, database, version = initialize()
    if args.command == "doctor":
        checks = [filesystem_check(root), database_check(database), *environment_checks()]
        for result in checks:
            print(f"{'OK' if result.ok else 'WARN'} {result.name}: {result.value}" + (f" ({result.warning})" if result.warning else ""))
        print(f"schema_version={version}")
    elif args.command == "export":
        conn = connect(database)
        try:
            repos = Repositories(conn)
            models = repos.registry_models()
            model = next((item for item in models if item["id"] == args.model_id), None) if args.model_id else next((item for item in models if item["lifecycle"] == "active"), None)
            if model is None or model["model_type"] != "conditional_flow" or model["status"] != "ready":
                raise ValueError("export requires a ready conditional-flow model")
            source = (root / model["manifest_relative_path"]).parent
            destination = (args.destination or (root / "exports" / f"{model['id']}-portable")).resolve()
            manifest = export_conditional_flow(source, destination)
            repos.audit("model", model["id"], "exported", {"destination": str(destination), "manifest": manifest})
            print(destination)
        finally:
            conn.close()
    elif args.command == "generate":
        def pair(value: str) -> tuple[float, float]:
            parts = [float(item) for item in value.split(",")]
            if len(parts) != 2: raise ValueError("coordinates must be x,y")
            return parts[0], parts[1]
        start, target = pair(args.start), pair(args.target)
        desktop = [float(item) for item in args.desktop.split(",")]
        if len(desktop) != 4: raise ValueError("desktop must be left,top,width,height")
        request = GenerationRequest(*start, *target, args.radius, *desktop, click_requested=args.click, random_seed=args.seed)
        runtime = OnnxFlowRuntime(args.model)
        result = runtime.generate(request)
        payload = {
            "schema_version": 1, "model_id": runtime.manifest["model_id"], "seed": args.seed,
            "request": {"start": start, "target": target, "radius": args.radius, "click": args.click},
            "reaction_delay_ns": 0, "movement_duration_ns": result.movement_duration_ns, "click_delay_ns": 0,
            "samples": [{"relative_time_ns": item.relative_time_ns, "x": item.x, "y": item.y} for item in result.samples],
        }
        args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return 0
