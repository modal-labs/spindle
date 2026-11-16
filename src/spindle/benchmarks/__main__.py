from __future__ import annotations

import argparse
import json
import os
import statistics
import time
import uuid
from pathlib import Path

import httpx
import uvicorn

from .runner import run
from .auth import modal_proxy_headers
from .schema import BenchmarkConfig
from .server import create_app
from .store import Store


def comparison(store, baseline, candidate, *, exploratory=False):
    bundles = [store.bundle(run_id) for run_id in (baseline, candidate)]
    manifests = [b["manifest"] for b in bundles]
    reasons = []
    for field in ("workload_fingerprint", "configuration_fingerprint"):
        if not manifests[0].get(field) or manifests[0].get(field) != manifests[1].get(
            field
        ):
            reasons.append(f"{field} missing or different")
    if manifests[0].get("recipe_fingerprint") != manifests[1].get("recipe_fingerprint"):
        reasons.append("recipe source/data fingerprint differs")
    if manifests[0].get("harness_source_sha256") != manifests[1].get(
        "harness_source_sha256"
    ) or manifests[0].get("packages") != manifests[1].get("packages"):
        reasons.append("benchmark harness or client packages differ")
    prompts = [
        {
            json.dumps(r.get("labels", {}), sort_keys=True): r["raw"].get(
                "prompt_sha256"
            )
            for r in bundle["records"]
            if r.get("kind") == "workload"
        }
        for bundle in bundles
    ]
    if prompts[0] != prompts[1]:
        reasons.append("tokenized workload prompts differ")
    if any(not m.get("revision_verified") for m in manifests):
        reasons.append("server revision not verified")
    if any(m.get("status") != "completed" for m in manifests):
        reasons.append("run not completed")
    hardware = []
    for manifest in manifests:
        if (
            not manifest.get("resolved_deployment", {})
            .get("recipe", {})
            .get("weights_revision")
        ):
            reasons.append("model weights revision not pinned")
        devices = {}
        for row in store.series([manifest["run_id"]], "spindle.gpu.memory.total"):
            labels = row["labels"]
            if labels.get("gpu.uuid"):
                devices[labels["gpu.uuid"]] = (
                    labels.get("spindle.component", "unknown"),
                    labels.get("gpu.name", "unknown"),
                    row["value"],
                )
        hardware.append(sorted(devices.values()))
    if not all(hardware):
        reasons.append("actual GPU observations missing")
    elif hardware[0] != hardware[1]:
        reasons.append("actual GPU allocations differ")
    reasons = list(dict.fromkeys(reasons))
    if reasons and not exploratory:
        raise ValueError(
            "Cannot make a controlled comparison: "
            + "; ".join(reasons)
            + ". Use --exploratory to inspect values with these limitations."
        )
    rows = []
    for metric in store.catalog([baseline, candidate]):
        groups = {}
        for row in store.series([baseline, candidate], metric["name"]):
            if (
                row["kind"] != "measurement"
                or row["labels"].get("phase") != "measurement"
            ):
                continue
            labels = {k: v for k, v in row["labels"].items() if k != "step"}
            key = json.dumps(labels, sort_keys=True)
            groups.setdefault(key, {}).setdefault(row["run_id"], []).append(
                row["value"]
            )
        for key, values in groups.items():
            left, right = values.get(baseline, []), values.get(candidate, [])
            if left and right:
                a, b = statistics.median(left), statistics.median(right)
                rows.append(
                    {
                        "metric": metric["name"],
                        "labels": json.loads(key),
                        "baseline_median": a,
                        "candidate_median": b,
                        "change_percent": (b / a - 1) * 100
                        if a and not reasons
                        else None,
                        "baseline_samples": len(left),
                        "candidate_samples": len(right),
                    }
                )
    return {
        "baseline": baseline,
        "candidate": candidate,
        "limitations": reasons,
        "metrics": rows,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Spindle benchmark runner, telemetry receiver, and local results browser"
    )
    parser.add_argument(
        "--directory", default=".spindle-benchmarks", help="Local result store"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    serve = commands.add_parser("serve")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8765)
    validate = commands.add_parser("validate")
    validate.add_argument("config")
    launch = commands.add_parser("run")
    launch.add_argument("config")
    launch.add_argument(
        "--revision", required=True, help="Committed Git revision to benchmark"
    )
    launch.add_argument("--root", type=Path, default=Path.cwd())
    launch.add_argument(
        "--base-url",
        help="Use an existing endpoint; revision will be marked unverified",
    )
    launch.add_argument(
        "--env", help="Modal environment for this run's isolated deployment"
    )
    launch.add_argument(
        "--otlp-endpoint", help="Collector base URL reachable from Modal containers"
    )
    ingest = commands.add_parser("ingest")
    ingest.add_argument("file", type=Path)
    upload = commands.add_parser("upload")
    upload.add_argument("file", type=Path)
    upload.add_argument("--url", required=True)
    upload.add_argument(
        "--modal-proxy-auth",
        action="store_true",
        help="Authenticate to the explicitly selected HTTPS Modal collector",
    )
    export = commands.add_parser("export")
    export.add_argument("run_id")
    export.add_argument("--output", required=True, type=Path)
    imported = commands.add_parser(
        "import-json",
        help="Inspect legacy result JSON; marked as an uncontrolled import",
    )
    imported.add_argument("file", type=Path)
    imported.add_argument("--name")
    compare = commands.add_parser("compare")
    compare.add_argument("baseline")
    compare.add_argument("candidate")
    compare.add_argument("--exploratory", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.command == "serve":
            token = os.getenv("SPINDLE_BENCHMARK_TOKEN")
            if args.host not in ("127.0.0.1", "localhost", "::1") and not token:
                raise ValueError(
                    "Set SPINDLE_BENCHMARK_TOKEN before exposing the server beyond loopback"
                )
            uvicorn.run(
                create_app(args.directory, token=token), host=args.host, port=args.port
            )
        elif args.command == "validate":
            config = BenchmarkConfig.model_validate_json(Path(args.config).read_text())
            print(config.model_dump_json(indent=2))
        elif args.command == "run":
            print(
                run(
                    args.config,
                    root=args.root,
                    directory=args.directory,
                    revision=args.revision,
                    base_url=args.base_url,
                    environment=args.env,
                    otlp_endpoint=args.otlp_endpoint,
                )
            )
        elif args.command == "upload":
            token = os.getenv("SPINDLE_BENCHMARK_TOKEN")
            response = httpx.post(
                args.url.rstrip("/") + "/api/upload",
                content=args.file.read_bytes(),
                headers={
                    "Content-Type": "application/json",
                    **(modal_proxy_headers(args.url) if args.modal_proxy_auth else {}),
                    **({"Authorization": "Bearer " + token} if token else {}),
                },
                timeout=60,
            )
            response.raise_for_status()
            print(response.text)
        else:
            store = Store(args.directory)
            if args.command == "ingest":
                bundle = json.loads(args.file.read_text())
                store.ingest(bundle)
                print(bundle["manifest"]["run_id"])
            elif args.command == "export":
                args.output.write_text(json.dumps(store.bundle(args.run_id), indent=2))
            elif args.command == "import-json":
                run_id = "import-" + uuid.uuid4().hex[:12]
                store.ingest(
                    {
                        "schema_version": 1,
                        "manifest": {
                            "run_id": run_id,
                            "name": args.name or args.file.stem,
                            "status": "imported",
                            "created_at": time.time(),
                            "revision_verified": False,
                        },
                        "records": [
                            {
                                "kind": "legacy-import",
                                "time": time.time(),
                                "metrics": json.loads(args.file.read_text()),
                                "labels": {"source": args.file.name},
                            }
                        ],
                    }
                )
                print(run_id)
            else:
                print(
                    json.dumps(
                        comparison(
                            store,
                            args.baseline,
                            args.candidate,
                            exploratory=args.exploratory,
                        ),
                        indent=2,
                    )
                )
    except (ValueError, OSError, KeyError, httpx.HTTPError) as exc:
        parser.exit(1, str(exc) + "\n")


if __name__ == "__main__":
    main()
