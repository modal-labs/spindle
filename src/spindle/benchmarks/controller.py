"""Run controllers with credentials supplied by the selected Modal environment."""

import json
import os
import subprocess
import tempfile
from pathlib import Path
from urllib.parse import quote

import httpx
import modal

from spindle.providers.modal.image_dependencies import (
    CORE_PACKAGES,
    STITCH_PACKAGE,
    TINKER_PACKAGE,
)

from .auth import modal_proxy_headers
from .runner import run
from .schema import BenchmarkConfig
from .store import Store

ROOT = Path(__file__).resolve().parents[3]
app = modal.App("spindle-training-benchmark-controller")
base_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install(
        *CORE_PACKAGES,
        STITCH_PACKAGE,
        TINKER_PACKAGE,
        "huggingface-hub",
        "numpy",
        "transformers",
        "questionary",
        "rich",
    )
)


def controller_image(worker=None):
    configured = base_image
    if worker:
        configured = configured.pip_install(
            "swegym @ git+https://github.com/SWE-Gym/SWE-Bench-Package.git@16dd480cce9b27bf111a362d280881c6def5d2a7"
        )
    configured = configured.env(
        {
            "PYTHONPATH": "/opt/spindle-benchmarks/src",
            **(
                {"SWE_GYM_WORKER": "/opt/spindle-swe-gym/" + worker.name}
                if worker
                else {}
            ),
        }
    )
    if worker:
        configured = configured.add_local_dir(worker.parent, "/opt/spindle-swe-gym")
    return configured.add_local_dir(
        ROOT / "src", "/opt/spindle-benchmarks/src"
    ).add_local_dir(ROOT / "examples", "/opt/spindle-benchmarks/examples")


worker_path = os.environ.get("SWE_GYM_WORKER")
image = controller_image(
    Path(worker_path) if worker_path and Path(worker_path).is_file() else None
)


@app.function(image=image, timeout=8 * 60 * 60)
def execute(
    config_json,
    archive,
    revision,
    environment,
    harness_revision,
    collector_app="spindle-benchmarks",
):
    root = Path("/opt/spindle-benchmarks")
    os.environ["SPINDLE_BENCHMARK_HARNESS_REVISION"] = harness_revision
    collector = modal.Function.from_name(
        collector_app, "server", environment_name=environment
    ).get_web_url()
    headers = modal_proxy_headers(collector)
    os.environ["OTEL_EXPORTER_OTLP_HEADERS"] = ",".join(
        k + "=" + quote(v, safe="") for k, v in headers.items()
    )
    with tempfile.TemporaryDirectory(
        prefix="spindle-benchmark-controller-"
    ) as temporary:
        directory = Path(temporary)
        config = directory / "config.json"
        config.write_text(config_json)
        failure = None
        try:
            run(
                config,
                root=root,
                directory=directory,
                revision=revision,
                environment=environment,
                otlp_endpoint=collector,
                source_archive=archive,
                results_url=collector,
                results_headers=headers,
            )
        except Exception as exc:
            failure = type(exc).__name__ + ": " + str(exc)
        store = Store(directory)
        bundle = store.bundle(store.runs()[0]["run_id"])
        response = httpx.post(
            collector + "/api/upload", headers=headers, json=bundle, timeout=120
        )
        response.raise_for_status()
        combined = httpx.get(
            collector + "/api/runs/" + bundle["manifest"]["run_id"] + "/bundle",
            headers=headers,
            timeout=120,
        )
        combined.raise_for_status()
        result = combined.json()
        if failure:
            result["manifest"]["controller_error"] = failure
        return result


@app.local_entrypoint()
def main(
    environment: str,
    config: str = "",
    revision: str = "",
    run_id: str = "",
    list_runs: bool = False,
    owner: str = "",
    cleanup_owner: str = "",
    directory: str = ".spindle-benchmarks",
    api_secret: str = "spindle-api",
    proxy_secret: str = "spindle-proxy",
    modal_secret: str = "",
    collector_app: str = "spindle-benchmarks",
):
    if cleanup_owner:
        result = stop_owned_runs.with_options(
            secrets=[modal.Secret.from_name(proxy_secret, environment_name=environment)]
        ).remote(environment, cleanup_owner, collector_app)
        print(json.dumps(result, indent=2))
        if any(row["cleanup"] == "failed" for row in result):
            raise RuntimeError("Some owned benchmark apps could not be stopped")
        return
    if list_runs or run_id:
        result = fetch_results.with_options(
            secrets=[modal.Secret.from_name(proxy_secret, environment_name=environment)]
        ).remote(environment, run_id, collector_app)
        if run_id:
            Store(directory).ingest(result)
            print(
                json.dumps(
                    {
                        "run_id": run_id,
                        "status": result["manifest"]["status"],
                        "records": len(result["records"]),
                    },
                    indent=2,
                )
            )
        else:
            print(
                json.dumps(
                    [
                        {
                            k: row.get(k)
                            for k in (
                                "run_id",
                                "name",
                                "status",
                                "revision",
                                "frontend",
                                "cleanup",
                            )
                        }
                        for row in result
                    ],
                    indent=2,
                )
            )
        return
    if not config or not revision:
        raise ValueError(
            "Select --config and --revision, or use --list-runs / --run-id"
        )
    config_json = Path(config).read_text()
    workload = BenchmarkConfig.model_validate_json(config_json)
    worker = None
    if workload.suite == "async-rl" and workload.recipe == "swe-gym":
        worker = Path(os.environ.get("SWE_GYM_WORKER", ""))
        if not worker.is_file() or not (worker.parent / "helpers").is_dir():
            raise ValueError(
                "Set SWE_GYM_WORKER to a pinned worker with its helpers directory"
            )
    sha = subprocess.check_output(
        ["git", "rev-parse", "--verify", revision + "^{commit}"], cwd=ROOT, text=True
    ).strip()
    archive = subprocess.check_output(["git", "archive", "--format=tar", sha], cwd=ROOT)
    harness = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    if subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT, text=True
    ).strip():
        harness += "-dirty"
    secrets = [
        modal.Secret.from_name(name, environment_name=environment)
        for name in (api_secret, proxy_secret, modal_secret, "huggingface-secret")
        if name
    ]
    bundle = execute.with_options(
        secrets=secrets, env={"SPINDLE_BENCHMARK_OWNER": owner}
    ).remote(config_json, archive, sha, environment, harness, collector_app)
    store = Store(directory)
    store.ingest(bundle)
    output = Path(directory) / (bundle["manifest"]["run_id"] + ".json")
    output.write_text(json.dumps(bundle, indent=2))
    print(
        json.dumps(
            {
                "run_id": bundle["manifest"]["run_id"],
                "status": bundle["manifest"]["status"],
                "result": str(output),
            },
            indent=2,
        )
    )
    if bundle["manifest"]["status"] != "completed":
        raise RuntimeError(
            bundle["manifest"].get(
                "controller_error", "Benchmark failed; see saved records"
            )
        )


@app.function(image=image, timeout=300)
def fetch_results(environment, run_id, collector_app="spindle-benchmarks"):
    collector = modal.Function.from_name(
        collector_app, "server", environment_name=environment
    ).get_web_url()
    path = "/api/runs/" + run_id + "/bundle" if run_id else "/api/runs"
    response = httpx.get(
        collector + path, headers=modal_proxy_headers(collector), timeout=120
    )
    response.raise_for_status()
    return response.json()


@app.function(image=image, timeout=900)
def stop_owned_runs(environment, owner, collector_app="spindle-benchmarks"):
    if not owner:
        raise ValueError("An exact benchmark owner ID is required")
    collector = modal.Function.from_name(
        collector_app, "server", environment_name=environment
    ).get_web_url()
    headers = modal_proxy_headers(collector)
    response = httpx.get(collector + "/api/runs", headers=headers, timeout=120)
    response.raise_for_status()
    result = []
    for manifest in response.json():
        frontend = manifest.get("frontend")
        if (
            manifest.get("owner") != owner
            or not isinstance(frontend, str)
            or not frontend.startswith("spindle-bench-")
        ):
            continue
        try:
            subprocess.run(
                [
                    "python",
                    "-m",
                    "spindle.deployment_cli",
                    "stop",
                    "--app",
                    frontend,
                    "--env",
                    environment,
                ],
                check=True,
                capture_output=True,
                timeout=300,
            )
            manifest["cleanup"] = "stopped"
            if manifest["status"] in ("starting", "running"):
                manifest["status"] = "cancelled"
        except Exception:
            manifest["cleanup"] = "failed"
        saved = httpx.post(
            collector + "/api/upload",
            headers=headers,
            json={"schema_version": 1, "manifest": manifest, "records": []},
            timeout=120,
        )
        saved.raise_for_status()
        result.append({"run_id": manifest["run_id"], "cleanup": manifest["cleanup"]})
    return result
