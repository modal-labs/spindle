"""Nightly synthetic monitoring for spindle configs, and on-demand regression runs.

Test it out with:

```shell
    uv run modal run -e spindle-ci -m synthetic_monitoring.config_validation::launch_nightly
```

Run tinker-cookbook's multi-turn guess-the-number GRPO recipe on one model, or on every
model the server serves, and compare with that model's last successful run:

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
import urllib.request
from dataclasses import asdict
from datetime import UTC, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

from synthetic_monitoring.chart import RunPoint, render_timing_history_chart

REPO_ROOT = Path(__file__).resolve().parent.parent
MODAL_ENV = "spindle-ci"
HISTORY_DICT_NAME = "spindle-synmon-prod-timings"
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
regression_image = with_local_sources(
    base_image.uv_pip_install("tinker-cookbook==0.6.1", "tinker==0.39.0")
)

slack_secret = modal.Secret.from_name("spindle-bot-slack", environment_name=MODAL_ENV)
api_secret = modal.Secret.from_name("spindle-prod-api", environment_name=MODAL_ENV)
hf_secret = modal.Secret.from_name("huggingface-secret", environment_name=MODAL_ENV)

app = modal.App("spindle-synmon-launcher")


def client_env(base_url: str) -> dict[str, str]:
    env = os.environ.copy()
    env["TINKER_BASE_URL"] = base_url
    env["PYTHONPATH"] = os.pathsep.join(
        filter(None, ["/opt/proxy-auth", env.get("PYTHONPATH")])
    )
    return env


def guess_number_renderer(model: str) -> str | None:
    # tinker-cookbook is installed only in the regression image.
    from tinker_cookbook.model_info import get_recommended_renderer_names

    try:
        names = get_recommended_renderer_names(model)
    except (KeyError, ValueError):
        return None
    for preferred in ("disable_thinking", "low_reasoning"):
        for name in names:
            if preferred in name:
                return name
    return names[0]


def run_guess_number(
    model: str, renderer: str, base_url: str
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
        command, check=True, env=client_env(base_url), timeout=GUESS_NUMBER["timeout_s"]
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


def _history_dict():
    return modal.Dict.from_name(
        HISTORY_DICT_NAME, create_if_missing=True, environment_name=MODAL_ENV
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
    config: str, point: RunPoint, one_success_per_day: bool = True
) -> list[RunPoint]:
    history = _history_dict()
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
        {k: v for k, v in asdict(p).items() if k != "metrics" or v} for p in points
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


@app.function(image=regression_image, secrets=[api_secret], timeout=10 * 60)
def served_models() -> dict[str, str | None]:
    """Each base model the server serves, with its guess-number renderer or None."""
    headers = {
        "x-api-key": os.environ["TINKER_API_KEY"],
        "Modal-Key": os.environ["MODAL_PROXY_TOKEN_ID"],
        "Modal-Secret": os.environ["MODAL_PROXY_TOKEN_SECRET"],
    }
    base_url = os.environ["TINKER_BASE_URL"].rstrip("/").removesuffix("/api/v1")
    request = urllib.request.Request(
        base_url + "/api/v1/get_server_capabilities", headers=headers
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        models = json.load(response)["supported_models"]
    return {
        m["model_name"]: guess_number_renderer(m["model_name"])
        for m in models
        if "/" in m["model_name"]
    }


@app.function(
    image=regression_image,
    timeout=GUESS_NUMBER["timeout_s"] + 15 * 60,
    secrets=[api_secret],
    single_use_containers=True,
)
def regression(model: str, renderer: str) -> dict:
    config = f"guess-number/{model}"
    started = time.time()
    error: str | None = None
    timings: dict[str, float] = {}
    metrics: dict[str, float] = {}
    try:
        timings, metrics = run_guess_number(
            model, renderer, os.environ["TINKER_BASE_URL"]
        )
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        print(f"error: {config} failed: {error}")
    point = RunPoint(
        ts=started,
        timings=timings,
        total_duration_s=time.time() - started,
        status="failed" if error else "success",
        metrics=metrics,
    )
    return _row(config, point, _record(config, point, one_success_per_day=False), error)


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
def regress(model: str = "", all_models: bool = False, parallel: int = 4) -> None:
    if bool(model) == all_models:
        raise SystemExit("Pass --model <base model>[,<base model>...] or --all-models.")
    runnable, skipped = select_runs(
        served_models.remote(),
        None if all_models else list(dict.fromkeys(model.split(","))),
    )
    for name, reason in skipped:
        print(f"skip   {name}: {reason}")
    runs = regression.with_options(max_containers=parallel).starmap(
        runnable, return_exceptions=True
    )
    failed = False
    for (name, _), row in zip(runnable, runs, strict=True):
        if isinstance(row, Exception):
            print(f"FAILED guess-number/{name}: {type(row).__name__}: {row}")
            failed = True
        else:
            _print_regression(row)
            failed |= row["status"] != "success"
    if failed:
        raise SystemExit(1)
