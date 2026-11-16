from __future__ import annotations

import asyncio
import ast
import contextlib
import hashlib
import importlib.metadata
import json
import os
import subprocess
import sys
import tarfile
import tempfile
import threading
import time
import uuid
from pathlib import Path

import httpx

from spindle.telemetry.benchmark import (
    provider as metrics_provider,
    record as emit_metrics,
)

from .schema import BenchmarkConfig
from .store import Store
from .workloads import control_plane_suite, command_suite, sdk_suite


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def package_version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "source-tree"


def source_pins(source):
    pins = {}
    for path in (source / "src/spindle/providers/modal").glob("*image*.py"):
        values = {}
        for statement in ast.parse(path.read_text()).body:
            if (
                isinstance(statement, ast.Assign)
                and len(statement.targets) == 1
                and isinstance(statement.targets[0], ast.Name)
            ):
                name = statement.targets[0].id
                if name.endswith(
                    ("_COMMIT", "_REVISION", "_IMAGE", "_PACKAGE", "_RELEASE")
                ):
                    try:
                        value = ast.literal_eval(statement.value)
                    except (ValueError, TypeError):
                        continue
                    if isinstance(value, str):
                        values[name] = value
        if values:
            pins[path.name] = values
    return pins


class Recorder:
    def __init__(
        self, store, manifest, output, *, results_url=None, results_headers=None
    ):
        self.store, self.manifest, self.output = store, manifest, output
        self.run_id = manifest["run_id"]
        self.store.register(manifest)
        self.results_url = results_url
        self.results_headers = results_headers or {}
        self.stop_upload = threading.Event()
        self.upload_thread = None
        if results_url:
            self.upload_thread = threading.Thread(
                target=self._publish_periodically, daemon=True
            )
            self.upload_thread.start()

    def _publish(self):
        if not self.results_url:
            return
        try:
            response = httpx.post(
                self.results_url.rstrip("/") + "/api/upload",
                headers=self.results_headers,
                json=self.store.bundle(self.run_id),
                timeout=10,
            )
            response.raise_for_status()
        except httpx.HTTPError as exc:
            self.collection_error("result upload", type(exc).__name__)

    def _publish_periodically(self):
        while not self.stop_upload.is_set():
            self._publish()
            self.stop_upload.wait(2)

    def close(self):
        self.stop_upload.set()
        if self.upload_thread is not None:
            self.upload_thread.join(timeout=12)
            self._publish()

    def record(self, metrics, labels=None, *, raw=None, stamp=None):
        labels = labels or {}
        self.store.append(
            self.run_id,
            {
                "kind": "measurement",
                "time": stamp or time.time(),
                "metrics": metrics,
                "labels": labels,
                **({"raw": raw} if raw is not None else {}),
            },
        )
        emit_metrics(
            metrics, {**labels, "spindle.run_id": self.run_id}, "spindle.benchmark"
        )

    def event(self, kind, raw, labels=None):
        self.store.append(
            self.run_id,
            {"kind": kind, "time": time.time(), "raw": raw, "labels": labels or {}},
        )

    def collection_error(self, source, message):
        self.store.append(
            self.run_id,
            {
                "kind": "collection-error",
                "time": time.time(),
                "source": source,
                "error": message,
                "metrics": {"collection.errors": 1},
            },
        )

    def finish(self, status, **fields):
        for log in self.output.glob("*.log"):
            content = log.read_text(errors="replace")[-64000:]
            for key, value in os.environ.items():
                if len(value) >= 8 and any(
                    word in key.upper()
                    for word in ("TOKEN", "SECRET", "PASSWORD", "API_KEY", "HEADERS")
                ):
                    content = content.replace(value, "[redacted]")
            self.store.append(
                self.run_id,
                {
                    "kind": "log",
                    "time": log.stat().st_mtime,
                    "labels": {"file": log.name},
                    "raw": {"text": content, "tail_bytes": 64000},
                },
            )
        self.manifest.update(status=status, finished_at=time.time(), **fields)
        self.store.register(self.manifest)
        path = self.output / "results.json"
        temporary = path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(self.store.bundle(self.run_id), indent=2, allow_nan=False)
        )
        temporary.replace(path)
        return path


def snapshot(root, revision, target, source_archive=None):
    archive = target / "source.tar"
    if source_archive is not None:
        archive.write_bytes(source_archive)
        with tarfile.open(archive) as tar:
            sha = tar.pax_headers.get("comment")
        if sha != revision:
            raise ValueError("Source archive commit does not match requested revision")
    else:
        sha = subprocess.check_output(
            ["git", "rev-parse", "--verify", revision + "^{commit}"],
            cwd=root,
            text=True,
        ).strip()
        with archive.open("wb") as stream:
            subprocess.run(
                ["git", "archive", "--format=tar", sha],
                cwd=root,
                stdout=stream,
                check=True,
            )
    source = target / "source"
    source.mkdir()
    with tarfile.open(archive) as tar:
        tar.extractall(source, filter="data")
    archive.unlink()
    return sha, source


DEPLOY_SCRIPT = """
import json, os, sys
import modal
from spindle.deployments import load, DeploymentConfig
from spindle.deployment_cli import deploy
settings=json.loads(os.environ["BENCHMARK_DEPLOY_SETTINGS"])
recipe=load(settings["recipe"])
for key,value in settings["overrides"].items():
    parts=key.split(".")
    if not hasattr(recipe,parts[0]): raise ValueError("Unknown recipe override: "+key)
    target=recipe.__dict__
    for part in parts[:-1]: target=target.setdefault(part,{})
    target[parts[-1]]=value
recipe.platform["frontend"]=settings["frontend"]
recipe.platform["modal"]["environment"]=settings["environment"]
recipe.platform.setdefault("requires_proxy_auth", True)
for role in ("trainer_env", "inference_env"):
    telemetry={**settings["telemetry"]}
    telemetry["OTEL_RESOURCE_ATTRIBUTES"] += ",spindle.component=" + ("trainer" if role=="trainer_env" else "sampler")
    setattr(recipe, role, {**getattr(recipe,role), **telemetry})
if settings["telemetry"].get("SPINDLE_BENCHMARK_METRICS") == "1":
    recipe.sglang_cfg={**{key:True for key in ("enable_metrics_for_all_schedulers","enable_forward_pass_metrics","enable_mfu_metrics","enable_cache_report","enable_request_time_stats_logging")},**recipe.sglang_cfg}
if settings["model"] not in (recipe.name,recipe.model): raise ValueError("Benchmark model does not match deployment recipe")
if settings["parameterization"] != recipe.parameterization: raise ValueError("Benchmark parameterization does not match deployment")
if settings["context_required"] > recipe.max_context_length: raise ValueError("Workload exceeds deployment context length")
config=DeploymentConfig.create(recipe)
with open(settings["resolved"],"w") as f: json.dump(config.model_dump(mode="json"),f,default=str)
deploy([config])
url=modal.Function.from_name(settings["frontend"],"server",environment_name=settings["environment"]).get_web_url()
with open(settings["url_file"],"w") as f: json.dump({"url":url},f)
"""


def safe_config(value):
    if isinstance(value, dict):
        return {
            k: "[redacted]"
            if any(
                word in k.lower()
                for word in ("header", "password", "api_key", "token_secret")
            )
            else safe_config(v)
            for k, v in value.items()
        }
    if isinstance(value, list):
        return [safe_config(v) for v in value]
    return value


def run(
    config_path,
    *,
    root,
    directory,
    revision,
    base_url=None,
    environment=None,
    otlp_endpoint=None,
    source_archive=None,
    results_url=None,
    results_headers=None,
):
    config = BenchmarkConfig.model_validate_json(Path(config_path).read_text())
    deadline = time.monotonic() + config.timeout_s
    run_id = time.strftime("b-%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:8]
    frontend = "spindle-bench-" + uuid.uuid4().hex[:12]
    output = Path(directory).resolve() / "artifacts" / run_id
    output.mkdir(parents=True)
    workload = config.model_dump(
        exclude={"deployment", "deployment_overrides", "name", "experiment"}
    )
    manifest = {
        "run_id": run_id,
        "name": config.name,
        "suite": config.suite,
        "config": safe_config(config.model_dump()),
        "created_at": time.time(),
        "status": "starting",
        "owner": os.getenv("SPINDLE_BENCHMARK_OWNER"),
        "frontend": frontend if not base_url else None,
        "environment": environment,
        "workload_fingerprint": digest(workload),
        "revision_verified": False,
        "initialization_seed": None,
        "packages": {
            p: package_version(p)
            for p in (
                "modal-spindle",
                "tinker",
                "opentelemetry-sdk",
                "transformers",
                "numpy",
                "httpx",
                "modal",
            )
        },
    }
    recorder = Recorder(
        Store(directory),
        manifest,
        output,
        results_url=results_url,
        results_headers=results_headers,
    )
    deployment_started = False
    subprocess_env = None
    original_env = dict(os.environ)
    temporary_directory = tempfile.TemporaryDirectory(prefix="spindle-benchmark-")
    try:
        if not os.getenv("TINKER_API_KEY"):
            raise ValueError("Set TINKER_API_KEY for the benchmark deployment")
        with contextlib.nullcontext(temporary_directory.name) as temporary:
            sha, source = snapshot(root, revision, Path(temporary), source_archive)
            manifest.update(
                revision=sha,
                source_pins=source_pins(source),
                harness_source_sha256=digest(
                    {
                        name: Path(__file__).with_name(name).read_text()
                        for name in ("runner.py", "schema.py", "workloads.py")
                    }
                ),
                source_lock_sha256=digest((source / "uv.lock").read_text()),
                harness_revision=os.getenv("SPINDLE_BENCHMARK_HARNESS_REVISION")
                or subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], cwd=root, text=True
                ).strip(),
            )
            telemetry = {k: v for k, v in os.environ.items() if k.startswith("OTEL_")}
            telemetry.update(
                OTEL_SERVICE_NAME="spindle-benchmarking",
                OTEL_RESOURCE_ATTRIBUTES=f"service.version={sha},deployment.environment.name=benchmark,spindle.run_id={run_id}",
                SPINDLE_BENCHMARK_METRICS="1",
            )
            if otlp_endpoint:
                telemetry["OTEL_EXPORTER_OTLP_ENDPOINT"] = otlp_endpoint
                for signal in ("TRACES", "METRICS"):
                    telemetry[f"OTEL_EXPORTER_OTLP_{signal}_ENDPOINT"] = (
                        otlp_endpoint.rstrip("/") + "/v1/" + signal.lower()
                    )
                    telemetry[f"OTEL_EXPORTER_OTLP_{signal}_HEADERS"] = telemetry.get(
                        "OTEL_EXPORTER_OTLP_HEADERS", ""
                    )
            if not (source / "src/spindle/telemetry/benchmark.py").is_file():
                telemetry.pop("SPINDLE_BENCHMARK_METRICS", None)
                recorder.collection_error(
                    "server instrumentation",
                    "This revision predates native benchmark collectors; only its existing OTLP signals can be collected",
                )
            os.environ.update(telemetry)
            if not any(
                telemetry.get(k)
                for k in (
                    "OTEL_EXPORTER_OTLP_ENDPOINT",
                    "OTEL_EXPORTER_OTLP_METRICS_ENDPOINT",
                )
            ):
                recorder.collection_error(
                    "otlp",
                    "No OTLP destination configured; only client observations are retained",
                )
            if base_url:
                recorder.event(
                    "coverage",
                    {
                        "warning": "External endpoint revision is asserted by the caller, not verified; deploy this source with benchmark telemetry enabled"
                    },
                )
            else:
                if not config.deployment or not environment:
                    raise ValueError(
                        "An isolated deployment requires config.deployment and --env; or supply --base-url"
                    )
                recipe = (source / config.deployment).resolve()
                if not recipe.is_relative_to(source):
                    raise ValueError("deployment must be a repository-relative path")
                subprocess_env = {
                    **os.environ,
                    "PYTHONPATH": str(source / "src"),
                    "SPINDLE_COMMIT": sha,
                    "MODAL_ENVIRONMENT": environment,
                    "BENCHMARK_DEPLOY_SETTINGS": json.dumps(
                        {
                            "recipe": str(recipe),
                            "model": config.model,
                            "parameterization": config.parameterization,
                            "context_required": max(config.context_lengths)
                            + (
                                config.output_tokens
                                if config.suite == "inference"
                                else 0
                            ),
                            "overrides": config.deployment_overrides,
                            "frontend": frontend,
                            "environment": environment,
                            "telemetry": telemetry,
                            "resolved": str(output / "resolved-private.json"),
                            "url_file": str(output / "url-private.json"),
                        }
                    ),
                }
                if subprocess_env.get("MODAL_TOKEN_ID") and subprocess_env.get(
                    "MODAL_TOKEN_SECRET"
                ):
                    # The standalone deployment child uses its explicit client credentials.
                    subprocess_env.pop("MODAL_IS_REMOTE", None)
                deployment_started = True
                with (output / "deployment.log").open("w") as log:
                    subprocess.run(
                        [sys.executable, "-c", DEPLOY_SCRIPT],
                        cwd=source,
                        env=subprocess_env,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        check=True,
                        timeout=max(0.1, deadline - time.monotonic()),
                    )
                resolved = json.loads((output / "resolved-private.json").read_text())
                comparable = dict(resolved["recipe"])
                comparable["platform"] = {"modal": comparable["platform"]["modal"]}
                for role in ("trainer_env", "inference_env"):
                    comparable[role] = {
                        k: v
                        for k, v in comparable.get(role, {}).items()
                        if not k.startswith(("OTEL_", "SPINDLE_BENCHMARK"))
                    }
                manifest.update(
                    resolved_deployment=safe_config(resolved),
                    configuration_fingerprint=digest(comparable),
                    revision_verified=True,
                    frontend=frontend,
                    environment=environment,
                )
                base_url = json.loads((output / "url-private.json").read_text())["url"]
            manifest["status"] = "running"
            recorder.store.register(manifest)
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("Benchmark exceeded its total execution budget")
            if config.suite == "async-rl":
                command_suite(
                    config.model_copy(update={"timeout_s": remaining}),
                    root,
                    output,
                    base_url,
                    recorder,
                )
            else:
                asyncio.run(
                    asyncio.wait_for(
                        (
                            control_plane_suite
                            if config.suite == "control-plane"
                            else sdk_suite
                        )(config, base_url, recorder),
                        timeout=remaining,
                    )
                )
            status = "completed"
    except BaseException as exc:
        status = "cancelled" if isinstance(exc, KeyboardInterrupt) else "failed"
        recorder.event("failure", {"type": type(exc).__name__, "message": str(exc)})
        recorder.finish(status)
        raise
    finally:
        if deployment_started:
            try:
                with (output / "cleanup.log").open("w") as log:
                    subprocess.run(
                        [
                            sys.executable,
                            "-m",
                            "spindle.deployment_cli",
                            "stop",
                            "--app",
                            frontend,
                            "--env",
                            environment,
                        ],
                        cwd=source,
                        env=subprocess_env,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        check=True,
                        timeout=300,
                    )
                manifest["cleanup"] = "stopped"
            except Exception as exc:
                manifest["cleanup"] = "failed"
                recorder.collection_error("cleanup", type(exc).__name__)
        for private in output.glob("*-private.json"):
            private.unlink()
        sink = metrics_provider()
        if sink is not None:
            try:
                sink.close()
            except Exception as exc:
                recorder.collection_error("metrics shutdown", type(exc).__name__)
        metrics_provider.cache_clear()
        for key in set(os.environ) - set(original_env):
            os.environ.pop(key, None)
        os.environ.update(original_env)
        temporary_directory.cleanup()
        recorder.finish(status)
        recorder.close()
    return recorder.finish(status)
