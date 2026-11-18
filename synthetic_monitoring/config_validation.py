"""Nightly synthetic monitoring for spindle configs, and on-demand regression runs.

Test it out with:

```shell
    uv run modal run -e spindle-ci -m synthetic_monitoring.config_validation::launch_nightly
```

Run tinker-cookbook's multi-turn guess-the-number GRPO recipe on one model, or on every
model with a checkout LoRA recipe. Deploys an isolated checkout server, then
runs the workloads against it and stops it on exit:

```shell
    uv run modal run -e spindle-ci -m synthetic_monitoring.config_validation::regress --model Qwen/Qwen3-8B
    uv run modal run -e spindle-ci -m synthetic_monitoring.config_validation::regress --all-models
```
"""

from __future__ import annotations

import json
import os
import statistics
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict
from datetime import UTC, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

from synthetic_monitoring.chart import RunPoint, render_timing_history_chart
from synthetic_monitoring.checkout import (
    checkout_server,
    regression_recipes,
    verify_checkout,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
MODAL_ENV = "spindle-ci"
HISTORY_DICT_NAME = "spindle-synmon-prod-timings"
REGRESSION_HISTORY_DICT_NAME = "spindle-synmon-checkout-timings"
NIGHTLY_TZ = "America/Los_Angeles"
TRAINER_APPS = {
    "dapo-math": "prod-trainer-prod-qwen38-27b-lora-64k",
    "swe-gym": "prod-trainer-prod-qwen35-9b-lora-128k",
}
DAPO_DIR = "/opt/dapo-math"
SWE_GYM = "/opt/modal-examples/06_gpu_and_ml/swe_gym/swe_gym.py"
SWE_GYM_REV = "a590d8b608e526771378dc1bbf0dcb8b03ca0309"

SCRIPTS = {
    "dapo-math": {
        "clients": 2,
        "steps": 5,
        "timeout_s": 2 * 60 * 60,
    },
    "swe-gym": {
        "clients": 2,
        "steps": 5,
        "groups": 1,
        "timeout_s": 6 * 60 * 60,
    },
}
GUESS_NUMBER = {
    "recipe": "tinker_cookbook.recipes.multiplayer_rl.guess_number.train",
    "steps": 3,
    "batch_size": 8,
    "group_size": 4,
    "timeout_s": 3 * 60 * 60,
}
MONITOR_TIMEOUT_S = max(script["timeout_s"] for script in SCRIPTS.values()) + 15 * 60
LAUNCH_TIMEOUT_S = MONITOR_TIMEOUT_S + 30 * 60

base_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .run_commands(
        "git clone --filter=blob:none --sparse"
        " https://github.com/modal-labs/modal-examples.git /opt/modal-examples"
        " && git -C /opt/modal-examples sparse-checkout set 06_gpu_and_ml/swe_gym"
        f" && git -C /opt/modal-examples checkout {SWE_GYM_REV}"
    )
    .uv_sync(uv_project_dir=str(REPO_ROOT), extra_options="--no-dev")
    .uv_pip_install(
        "slack-sdk==3.27.1",
        "matplotlib==3.10.1",
        "transformers[chat-template]==5.17.0",
        "swegym @ git+https://github.com/SWE-Gym/SWE-Bench-Package.git@16dd480cce9b27bf111a362d280881c6def5d2a7",
    )
    .env({"MODAL_ENVIRONMENT": MODAL_ENV})
)


def with_local_sources(image: modal.Image) -> modal.Image:
    return (
        image.add_local_python_source("spindle", "synthetic_monitoring")
        .add_local_dir(
            str(REPO_ROOT / "examples/dapo-27b-4h200-12b200"),
            remote_path=DAPO_DIR,
        )
        .add_local_file(
            str(REPO_ROOT / "examples/shared-server/with_proxy_auth.py"),
            "/opt/proxy-auth/with_proxy_auth.py",
        )
        .add_local_file(
            str(REPO_ROOT / "synthetic_monitoring/sitecustomize.py"),
            "/opt/proxy-auth/sitecustomize.py",
        )
    )


probe_image = with_local_sources(base_image)
# The nightly clients keep the locked Tinker SDK; only regression runs use the cookbook.
# Work around thinking-machines-lab/tinker-cookbook#1017 until the next release.
regression_image = with_local_sources(
    base_image.uv_pip_install("tinker-cookbook==0.6.1", "tinker==0.39.0")
    .add_local_file(
        str(REPO_ROOT / "synthetic_monitoring/guess_number.patch"),
        "/opt/guess_number.patch",
        copy=True,
    )
    .run_commands(
        "python -c 'import site, subprocess; "
        'subprocess.run(["git", "-C", site.getsitepackages()[0], '
        '"apply", "/opt/guess_number.patch"], check=True)\''
    )
)

slack_secret = modal.Secret.from_name("spindle-bot-slack", environment_name=MODAL_ENV)
api_secret = modal.Secret.from_name("spindle-prod-api", environment_name=MODAL_ENV)
hf_secret = modal.Secret.from_name("huggingface-secret", environment_name=MODAL_ENV)

app = modal.App("spindle-synmon-launcher")


def client_env(base_url: str, api_key: str | None = None) -> dict[str, str]:
    env = os.environ.copy()
    env["TINKER_BASE_URL"] = base_url
    env["PYTHONPATH"] = os.pathsep.join(
        filter(None, ["/opt/proxy-auth", env.get("PYTHONPATH")])
    )
    if api_key is not None:
        env["TINKER_API_KEY"] = api_key
    return env


def guess_number_renderer(model: str) -> str | None:
    # tinker-cookbook is installed only in the regression image.
    from tinker_cookbook.model_info import get_recommended_renderer_names

    try:
        names = get_recommended_renderer_names(model)
    except (KeyError, ValueError):
        # Cookbook 0.6.1 omits these models; their existing renderers apply.
        return {
            "Qwen/Qwen2.5-3B": "role_colon",
            "Qwen/Qwen2.5-7B-Instruct": "qwen3_instruct",
            "Qwen/Qwen3-4B": "qwen3_disable_thinking",
            "Qwen/Qwen3.5-2B": "qwen3_5_disable_thinking",
        }.get(model)
    for preferred in ("disable_thinking", "low_reasoning"):
        for name in names:
            if preferred in name:
                return name
    return names[0]


def run_guess_number(
    model: str, renderer: str, base_url: str, api_key: str
) -> tuple[dict[str, float], dict[str, float]]:
    log_path = Path(tempfile.mkdtemp(prefix="synmon-guess-number-"))
    command = [
        sys.executable,
        "-m",
        GUESS_NUMBER["recipe"],
        f"model_name={model}",
        f"renderer_name={renderer}",
        f"max_steps={GUESS_NUMBER['steps']}",
        f"batch_size={GUESS_NUMBER['batch_size']}",
        f"group_size={GUESS_NUMBER['group_size']}",
        f"log_path={log_path}",
        "behavior_if_log_dir_exists=delete",
        "eval_every=0",
        "save_every=0",
    ]
    print("running", " ".join(command))
    started = time.time()
    subprocess.run(
        command,
        check=True,
        env=client_env(base_url, api_key),
        timeout=GUESS_NUMBER["timeout_s"],
    )
    lines = (log_path / "metrics.jsonl").read_text().splitlines()
    steps = [json.loads(line) for line in lines if line.strip()]
    steps = [step for step in steps if "env/all/reward/total" in step]
    if len(steps) != GUESS_NUMBER["steps"]:
        raise ValueError(f"Expected {GUESS_NUMBER['steps']} steps, got {len(steps)}")
    step_s = [step["time/total"] for step in steps]
    timings = {
        "startup": time.time() - started - sum(step_s),
        "first step": step_s[0],
        "warm steps": sum(step_s[1:]),
    }
    metrics = {
        "kl_sample_train_v2": statistics.mean(
            step["optim/kl_sample_train_v2"] for step in steps
        ),
        "reward_first": steps[0]["env/all/reward/total"],
        "reward_last": steps[-1]["env/all/reward/total"],
    }
    return timings, metrics


def run_script(name: str, base_url: str) -> dict[str, float]:
    spec = SCRIPTS[name]
    output = Path(tempfile.mkdtemp(prefix=f"synmon-{name}-"))
    if name == "dapo-math":
        work = output / "dapo"
        shutil.copytree(DAPO_DIR, work)
        cfg = json.loads((work / "config.json").read_text())
        cfg.update(
            updates=spec["steps"],
            context_tokens=8192,
            max_tokens=1024,
            groups_per_update=1,
            samples_per_group=4,
            max_candidate_groups=4,
            warmup_samples=1,
        )
        config = work / "config.json"
        config.write_text(json.dumps(cfg))
        command = [
            sys.executable,
            str(work / "launch_clients.py"),
            "--clients",
            str(spec["clients"]),
            "--mode",
            "gate",
            "--stagger-seconds",
            "10",
            "--config",
            str(config),
            "--base-url",
            base_url,
        ]
    else:
        command = [
            sys.executable,
            SWE_GYM,
            "--clients",
            str(spec["clients"]),
            "--steps",
            str(spec["steps"]),
            "--groups-per-batch",
            str(spec["groups"]),
            "--hours",
            str(spec["timeout_s"] / 3600),
            "--output",
            str(output),
        ]
    print("running", " ".join(command))
    started = time.time()
    subprocess.run(
        command, check=True, env=client_env(base_url), timeout=spec["timeout_s"]
    )
    clients: dict[int | str, list[dict]] = {}
    for path in output.rglob("events.jsonl"):
        for line in path.read_text().splitlines():
            event = json.loads(line)
            key = event.get("job", str(path))
            clients.setdefault(key, []).append(event)
    if len(clients) != spec["clients"]:
        raise ValueError(
            f"Expected events for {spec['clients']} clients, got {len(clients)}"
        )
    ready, first, last = [], [], []
    ready_event = "ready" if name == "dapo-math" else "rollout_start"
    step_event = "update_done" if name == "dapo-math" else "step_done"
    for events in clients.values():
        ready.append(next(e["time"] for e in events if e["event"] == ready_event))
        steps = sorted(e["time"] for e in events if e["event"] == step_event)
        if len(steps) != spec["steps"]:
            raise ValueError(f"Expected {spec['steps']} steps, got {len(steps)}")
        first.append(steps[0])
        last.append(steps[-1])
    return {
        "startup": max(ready) - started,
        "first step": max(first) - max(ready),
        "warm steps": max(last) - max(first),
    }


def _history_dict(name: str = HISTORY_DICT_NAME):
    return modal.Dict.from_name(
        name, create_if_missing=True, environment_name=MODAL_ENV
    )


def _raw_cell(text: str) -> dict:
    return {"type": "raw_text", "text": text}


def slack_report_blocks(rows: list[dict]) -> list[dict]:
    day = datetime.now(UTC).strftime("%Y-%m-%d")
    table_rows = [
        [
            _raw_cell("Script"),
            _raw_cell("Total runtime"),
            _raw_cell("Delta vs previous successful run"),
        ]
    ]
    for row in rows:
        if row["status"] == "success":
            runtime, delta = (
                _raw_cell(str(timedelta(seconds=round(row["total_duration_s"])))),
                _raw_cell(row["delta"]),
            )
        else:
            runtime, delta = _raw_cell("FAILED"), _raw_cell("n/a")
        table_rows.append([_raw_cell(row["config"]), runtime, delta])
    return [
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": f"Nightly validation - {day}",
            },
        },
        {
            "type": "table",
            "column_settings": [
                {"is_wrapped": True},
                {"align": "left"},
                {"is_wrapped": True},
            ],
            "rows": table_rows,
        },
    ]


def _record(
    config: str,
    point: RunPoint,
    one_success_per_day: bool = True,
    history_name: str = HISTORY_DICT_NAME,
) -> list[RunPoint]:
    history = _history_dict(history_name)
    points = [RunPoint(**item) for item in history.get(config, [])]
    if one_success_per_day and point.status == "success":
        day = datetime.fromtimestamp(point.ts, tz=ZoneInfo(NIGHTLY_TZ)).date()
        points = [
            p
            for p in points
            if p.status != "success"
            or datetime.fromtimestamp(p.ts, tz=ZoneInfo(NIGHTLY_TZ)).date() != day
        ]
    points.append(point)
    history[config] = [
        {k: v for k, v in asdict(p).items() if v is not None and (k != "metrics" or v)}
        for p in points
    ]
    return points


def _row(
    config: str, point: RunPoint, history: list[RunPoint], error: str | None = None
) -> dict:
    ok = point.status == "success"
    prior = next((p for p in reversed(history[:-1]) if p.status == "success"), None)
    delta = "n/a"
    if ok and prior:
        delta = ", ".join(
            f"{stage}: {seconds - prior.timings[stage]:+.0f}s"
            for stage, seconds in point.timings.items()
        )
    return {
        "config": config,
        "status": point.status,
        "total_duration_s": point.total_duration_s if ok else None,
        "delta": delta,
        "timings": point.timings,
        "metrics": point.metrics,
        "prior_metrics": prior.metrics if ok and prior else {},
        "modal_app_url": point.modal_app_url,
        "error": error,
    }


def _post_report(rows: list[dict]) -> None:
    from slack_sdk import WebClient
    from slack_sdk.errors import SlackApiError

    client = WebClient(token=os.environ["SLACK_BOT_TOKEN"])
    channel = os.environ["SLACK_CHANNEL_ID"]

    header = f"Nightly validation - {datetime.now(UTC):%Y-%m-%d}"
    thread_ts = client.chat_postMessage(
        channel=channel,
        text=header,
        blocks=slack_report_blocks(rows),
        mrkdwn=True,
        unfurl_links=False,
        unfurl_media=False,
    )["ts"]
    for row in rows:
        try:
            blocks = []
            text = ""
            if row["status"] == "success":
                png = render_timing_history_chart(
                    [
                        RunPoint(**item)
                        for item in _history_dict().get(row["config"], [])
                    ],
                    config_name=row["config"],
                )
                if not png:
                    continue
                uploaded = client.files_upload_v2(
                    content=png,
                    filename=f"timing_history_{row['config']}.png",
                    title=row["config"],
                )
                blocks.append(
                    {
                        "type": "image",
                        "slack_file": {"id": uploaded["files"][0]["id"]},
                        "alt_text": f"{row['config']} timing history",
                    }
                )
            else:
                text = f"`{row['config']}` failed"
                if row.get("error"):
                    text += f"\n```{row['error'][:1500]}```"
            if row.get("modal_app_url"):
                if text:
                    text += "\n"
                text += f"<{row['modal_app_url']}|Trainer logs>"
            if text:
                blocks.append(
                    {"type": "section", "text": {"type": "mrkdwn", "text": text}}
                )
            for attempt in range(5):
                try:
                    client.chat_postMessage(
                        channel=channel,
                        thread_ts=thread_ts,
                        text=text,
                        blocks=blocks,
                        mrkdwn=True,
                        unfurl_links=False,
                        unfurl_media=False,
                    )
                    break
                except SlackApiError as exc:
                    if attempt == 4 or not any(
                        "invalid slack file" in error
                        for error in exc.response.get("errors", [])
                    ):
                        raise
                    time.sleep(2**attempt)
        except Exception as exc:
            print(f"error: reply failed for {row['config']}: {exc}")


@app.function(
    image=probe_image,
    timeout=MONITOR_TIMEOUT_S,
    secrets=[api_secret, hf_secret],
    single_use_containers=True,
)
def monitor(script: str) -> dict:
    started = time.time()
    error: str | None = None
    timings = {}
    try:
        timings = run_script(script, os.environ["TINKER_BASE_URL"])
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        print(f"error: {script} failed: {error}")

    try:
        url = modal.App.lookup(
            TRAINER_APPS[script], environment_name="spindle"
        ).get_dashboard_url()
        url += f"?activeTab=logs&start={started}&end={time.time()}&live=false"
    except Exception:
        url = None
    point = RunPoint(
        ts=started,
        timings=timings,
        total_duration_s=time.time() - started,
        status="failed" if error else "success",
        modal_app_url=url,
    )
    return _row(script, point, _record(script, point), error)


def _failed_row(script: str, started: float, error: str) -> dict:
    point = RunPoint(
        ts=started,
        timings={},
        total_duration_s=max(0, time.time() - started),
        status="failed",
    )
    try:
        history = _record(script, point)
    except Exception as exc:
        print(f"error: recording {script} failed: {exc}")
        history = [point]
    return _row(script, point, history, error)


@app.function(
    image=probe_image,
    timeout=LAUNCH_TIMEOUT_S,
    schedule=modal.Cron("0 2 * * *", timezone=NIGHTLY_TZ),
    secrets=[slack_secret],
)
def launch_nightly() -> list[dict]:
    rows: list[dict] = []
    started = time.time()
    payloads = list(monitor.map(list(SCRIPTS), return_exceptions=True))
    for script, payload in zip(SCRIPTS, payloads):
        if isinstance(payload, Exception):
            error = f"{type(payload).__name__}: {payload}"
            print(f"error: {script} failed: {error}")
            rows.append(_failed_row(script, started, error))
        else:
            rows.append(payload)
    _post_report(rows)
    if any(row["status"] != "success" for row in rows):
        raise RuntimeError("synmon finished with failures")
    return rows


@app.function(image=regression_image, timeout=10 * 60)
def regression_renderers(models: list[str]) -> dict[str, str | None]:
    return {model: guess_number_renderer(model) for model in models}


@app.function(image=regression_image, timeout=3 * 60)
def verify_regression_server(
    base_url: str, api_key: str, commit: str, recipes: list[str]
) -> None:
    verify_checkout(base_url, api_key, commit, recipes)


@app.function(
    image=regression_image,
    timeout=GUESS_NUMBER["timeout_s"] + 15 * 60,
    single_use_containers=True,
)
def regression(
    model: str,
    recipe_name: str,
    renderer: str,
    base_url: str,
    api_key: str,
    commit: str,
) -> dict:
    config = f"guess-number/{recipe_name}"
    started = time.time()
    error: str | None = None
    timings: dict[str, float] = {}
    metrics: dict[str, float] = {}
    try:
        timings, metrics = run_guess_number(model, renderer, base_url, api_key)
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        print(f"error: {config} failed: {error}")
    point = RunPoint(
        ts=started,
        timings=timings,
        total_duration_s=time.time() - started,
        status="failed" if error else "success",
        metrics=metrics,
        source_commit=commit,
        server_url=base_url,
    )
    return _row(
        config,
        point,
        _record(
            config,
            point,
            one_success_per_day=False,
            history_name=REGRESSION_HISTORY_DICT_NAME,
        ),
        error,
    )


def select_runs(
    renderers: dict[str, str | None], models: list[str] | None
) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """Split the requested models, or every served model, into runs and skips."""
    runnable, skipped = [], []
    for name in renderers if models is None else models:
        renderer = renderers.get(name)
        if renderer is not None:
            runnable.append((name, renderer))
        elif name in renderers:
            skipped.append((name, "no cookbook renderer"))
        else:
            skipped.append((name, "not served"))
    return runnable, skipped


def _print_regression(row: dict) -> None:
    if row["status"] != "success":
        print(f"FAILED {row['config']}: {row['error']}")
        return
    prior = row["prior_metrics"]
    metrics = ", ".join(
        f"{key} {prior[key]:.4g} -> {value:.4g}"
        if key in prior
        else f"{key} {value:.4g}"
        for key, value in row["metrics"].items()
    )
    print(f"ok     {row['config']}: {metrics}; timing delta {row['delta']}")


@app.local_entrypoint()
def regress(
    model: str = "", all_models: bool = False, parallel: int = 4, config: str = ""
) -> None:
    if sum((bool(model), all_models, bool(config))) != 1:
        raise SystemExit(
            "Pass --model <base model>[,...], --config <recipe>[,...], or --all-models."
        )
    if parallel < 1:
        raise SystemExit("--parallel must be positive")
    recipes = regression_recipes(
        model.split(",") if model else None, config.split(",") if config else None
    )
    renderers = regression_renderers.remote(
        list(dict.fromkeys(recipe.model for recipe in recipes))
    )
    runnable, skipped = select_runs(renderers, None)
    for name, reason in skipped:
        print(f"skip   {name}: {reason}")
    supported = {name for name, _ in runnable}
    recipes = [recipe for recipe in recipes if recipe.model in supported]
    if not recipes:
        raise SystemExit("No selected recipes have a cookbook renderer")
    failed = False
    with checkout_server(recipes) as (base_url, api_key, commit, secret_name):
        credentials = modal.Secret.from_name(secret_name, environment_name=MODAL_ENV)
        verify_regression_server.with_options(secrets=[credentials]).remote(
            base_url, api_key, commit, [recipe.name for recipe in recipes]
        )
        worker = regression.with_options(max_containers=parallel, secrets=[credentials])
        calls = []
        pending = set()
        try:
            for recipe in recipes:
                print(f"Running {recipe.name}: {recipe.model}", flush=True)
                call = worker.spawn(
                    recipe.model,
                    recipe.name,
                    renderers[recipe.model],
                    base_url,
                    api_key,
                    commit,
                )
                calls.append((recipe.name, call))
                pending.add(call)
            for name, call in calls:
                try:
                    row = call.get()
                except Exception as exc:
                    print(f"FAILED guess-number/{name}: {type(exc).__name__}: {exc}")
                    failed = True
                else:
                    _print_regression(row)
                    failed |= row["status"] != "success"
                pending.remove(call)
        finally:
            for call in pending:
                call.cancel(terminate_containers=True)
    if failed:
        raise SystemExit(1)
