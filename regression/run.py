"""Run tinker-cookbook's multi-turn GRPO recipe against a Spindle server and record the results.

uv run --with tinker-cookbook==0.6.1 python regression/run.py --model Qwen/Qwen3-8B
uv run --with tinker-cookbook==0.6.1 python regression/run.py --all
"""

import argparse
import json
import os
import statistics
import subprocess
import sys
import tempfile
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

from tinker_cookbook.model_info import get_recommended_renderer_names

REPO = Path(__file__).resolve().parent.parent
RESULTS = REPO / "regression" / "results" / "guess_number.jsonl"
PROXY_AUTH = REPO / "examples" / "shared-server" / "with_proxy_auth.py"
RECIPE = "tinker_cookbook.recipes.multiplayer_rl.guess_number.train"
SETTINGS = ("model", "renderer", "steps", "batch_size", "group_size", "tinker_cookbook")
COMPARED = ("kl_v2", "median_step_s", "reward_last")


def server_info():
    headers = {"x-api-key": os.environ["TINKER_API_KEY"]}
    if os.environ.get("MODAL_PROXY_TOKEN_ID"):
        headers["Modal-Key"] = os.environ["MODAL_PROXY_TOKEN_ID"]
        headers["Modal-Secret"] = os.environ["MODAL_PROXY_TOKEN_SECRET"]
    base_url = os.environ["TINKER_BASE_URL"].rstrip("/").removesuffix("/api/v1")
    request = urllib.request.Request(
        base_url + "/api/v1/get_server_capabilities", headers=headers
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.load(response)


def renderer_for(model):
    names = get_recommended_renderer_names(model)
    for preferred in ("disable_thinking", "low_reasoning"):
        for name in names:
            if preferred in name:
                return name
    return names[0]


def run(model, renderer, args):
    log_path = Path(tempfile.mkdtemp(prefix="spindle-regression-"))
    launcher = [str(PROXY_AUTH)] if os.environ.get("MODAL_PROXY_TOKEN_ID") else []
    command = [
        sys.executable,
        *launcher,
        "-m",
        RECIPE,
        f"model_name={model}",
        f"renderer_name={renderer}",
        f"max_steps={args.steps}",
        f"batch_size={args.batch_size}",
        f"group_size={args.group_size}",
        f"log_path={log_path}",
        "behavior_if_log_dir_exists=delete",
        "eval_every=0",
        "save_every=0",
    ]
    if args.wandb_project:
        command.append(f"wandb_project={args.wandb_project}")
    row = {
        "model": model,
        "renderer": renderer,
        "steps": args.steps,
        "batch_size": args.batch_size,
        "group_size": args.group_size,
    }
    started = time.monotonic()
    error = ""
    try:
        done = subprocess.run(
            command, capture_output=True, text=True, timeout=args.timeout_s
        )
        if done.returncode:
            error = f"exit {done.returncode}: {done.stderr[-1500:]}"
    except subprocess.TimeoutExpired:
        error = f"timed out after {args.timeout_s} s"
    steps = []
    try:
        lines = (log_path / "metrics.jsonl").read_text().splitlines()
        steps = [json.loads(line) for line in lines if line.strip()]
    except (OSError, ValueError) as exc:
        error = error or f"could not read metrics: {exc}"
    steps = [step for step in steps if "env/all/reward/total" in step]
    row["wall_s"] = round(time.monotonic() - started)
    row["ok"] = not error and len(steps) == args.steps
    if steps:
        step_times = [step["time/total"] for step in steps if "time/total" in step]
        row["first_step_s"] = round(step_times[0], 1) if step_times else None
        row["median_step_s"] = (
            round(statistics.median(step_times[1:]), 1) if len(step_times) > 1 else None
        )
        row["reward_first"] = round(steps[0]["env/all/reward/total"], 3)
        row["reward_last"] = round(steps[-1]["env/all/reward/total"], 3)
        row["kl_v2"] = round(
            statistics.mean(s["optim/kl_sample_train_v2"] for s in steps), 5
        )
    return row, error


def previous(row):
    if not RESULTS.exists():
        return None
    same = [
        old
        for old in map(json.loads, RESULTS.read_text().splitlines())
        if old["ok"] and all(old.get(k) == row.get(k) for k in SETTINGS)
    ]
    return same[-1] if same else None


def report(row, error):
    old = previous(row)
    cells = []
    for key in COMPARED:
        before = old.get(key) if old else None
        now = row.get(key)
        cells.append(
            f"{key} {before} → {now}" if before is not None else f"{key} {now}"
        )
    print(
        f"{'ok' if row['ok'] else 'FAILED':6} {row['model']}: " + ", ".join(cells),
        flush=True,
    )
    if error:
        print(error, flush=True)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument(
        "--model", action="append", help="base model; repeat for several"
    )
    target.add_argument(
        "--all", action="store_true", help="every model the server serves"
    )
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--parallel", type=int, default=4)
    parser.add_argument("--timeout-s", type=int, default=3 * 60 * 60)
    parser.add_argument("--wandb-project")
    parser.add_argument("--no-save", action="store_true", help="print results only")
    args = parser.parse_args()
    if not os.environ.get("TINKER_BASE_URL"):
        parser.error("set TINKER_BASE_URL to the Spindle server")

    served = [
        m["model_name"]
        for m in server_info()["supported_models"]
        if "/" in m["model_name"]
    ]
    runnable = {}
    for model in served if args.all else args.model:
        try:
            runnable[model] = renderer_for(model)
        except (KeyError, ValueError) as exc:
            print(
                f"skip   {model}: tinker-cookbook has no renderer for it ({exc})",
                flush=True,
            )

    tools = {"tinker": version("tinker"), "tinker_cookbook": version("tinker-cookbook")}
    all_ok = True
    with ThreadPoolExecutor(args.parallel) as pool:
        futures = [
            pool.submit(run, model, renderer, args)
            for model, renderer in runnable.items()
        ]
        for future in as_completed(futures):
            row, error = future.result()
            row |= tools
            report(row, error)
            all_ok &= row["ok"]
            if args.no_save:
                continue
            info = server_info()
            stamp = {
                "date": datetime.now(UTC).isoformat(timespec="seconds"),
                "spindle_version": info.get("spindle_version"),
                "spindle_commit": info.get("spindle_commit"),
            }
            RESULTS.parent.mkdir(parents=True, exist_ok=True)
            with RESULTS.open("a") as handle:
                handle.write(json.dumps(stamp | row) + "\n")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
