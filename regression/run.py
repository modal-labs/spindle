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
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

from tinker_cookbook.model_info import get_recommended_renderer_names

REPO = Path(__file__).resolve().parent.parent
RESULTS = REPO / "regression" / "results" / "guess_number.jsonl"
PROXY_AUTH = REPO / "examples" / "shared-server" / "with_proxy_auth.py"
RECIPE = "tinker_cookbook.recipes.multiplayer_rl.guess_number.train"
COMPARED = ("kl_v2", "median_step_s", "reward_last")


def server_info(base_url):
    headers = {"x-api-key": os.environ["TINKER_API_KEY"]}
    if os.environ.get("MODAL_PROXY_TOKEN_ID"):
        headers["Modal-Key"] = os.environ["MODAL_PROXY_TOKEN_ID"]
        headers["Modal-Secret"] = os.environ["MODAL_PROXY_TOKEN_SECRET"]
    url = (
        base_url.rstrip("/").removesuffix("/api/v1") + "/api/v1/get_server_capabilities"
    )
    with urllib.request.urlopen(
        urllib.request.Request(url, headers=headers), timeout=60
    ) as response:
        return json.load(response)


def renderer_for(model):
    """Short answers keep each turn small, so prefer a renderer without thinking."""
    names = get_recommended_renderer_names(model)
    for preferred in ("disable_thinking", "low_reasoning"):
        for name in names:
            if preferred in name:
                return name
    return names[0]


def run(model, renderer, args, base_url):
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
    started = time.monotonic()
    try:
        done = subprocess.run(
            command,
            env={**os.environ, "TINKER_BASE_URL": base_url},
            capture_output=True,
            text=True,
            timeout=args.timeout_s,
        )
        error = done.stderr[-2000:] if done.returncode else ""
    except subprocess.TimeoutExpired:
        error = f"timed out after {args.timeout_s} s"
    steps = []
    metrics = log_path / "metrics.jsonl"
    if metrics.exists():
        steps = [json.loads(line) for line in metrics.read_text().splitlines()]
        steps = [step for step in steps if "env/all/reward/total" in step]
    return summarize(model, renderer, args, steps, time.monotonic() - started, error)


def summarize(model, renderer, args, steps, wall_s, error):
    row = {
        "model": model,
        "renderer": renderer,
        "steps": args.steps,
        "batch_size": args.batch_size,
        "group_size": args.group_size,
        "ok": not error and len(steps) == args.steps,
        "wall_s": round(wall_s),
    }
    if steps:
        step_times = [step["time/total"] for step in steps if "time/total" in step]
        row |= {
            "first_step_s": round(step_times[0], 1) if step_times else None,
            "median_step_s": round(statistics.median(step_times[1:]), 1)
            if len(step_times) > 1
            else None,
            "reward_first": round(steps[0]["env/all/reward/total"], 3),
            "reward_last": round(steps[-1]["env/all/reward/total"], 3),
            "kl_v1": round(
                statistics.mean(s["optim/kl_sample_train_v1"] for s in steps), 5
            ),
            "kl_v2": round(
                statistics.mean(s["optim/kl_sample_train_v2"] for s in steps), 5
            ),
        }
    if error:
        row["error"] = error
    return row


def previous(row):
    if not RESULTS.exists():
        return None
    same = [
        old
        for old in map(json.loads, RESULTS.read_text().splitlines())
        if old["ok"]
        and all(
            old[k] == row[k] for k in ("model", "steps", "batch_size", "group_size")
        )
    ]
    return same[-1] if same else None


def report(row):
    old = previous(row)
    cells = []
    for key in COMPARED:
        now = row.get(key)
        before = old.get(key) if old else None
        cells.append(
            f"{key} {before} → {now}" if before is not None else f"{key} {now}"
        )
    status = "ok" if row["ok"] else "FAILED"
    print(f"{status:6} {row['model']}: " + ", ".join(cells), flush=True)
    if not row["ok"]:
        print(row.get("error", "")[-600:], flush=True)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument(
        "--model", action="append", help="base model to run; repeat for several"
    )
    target.add_argument(
        "--all", action="store_true", help="every model the server serves"
    )
    parser.add_argument("--base-url", default=os.environ.get("TINKER_BASE_URL"))
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--parallel", type=int, default=4)
    parser.add_argument("--timeout-s", type=int, default=3 * 60 * 60)
    parser.add_argument("--wandb-project")
    parser.add_argument(
        "--no-save", action="store_true", help="print results without appending them"
    )
    args = parser.parse_args()
    if not args.base_url:
        parser.error("set --base-url or TINKER_BASE_URL")

    info = server_info(args.base_url)
    served = [
        m["model_name"] for m in info["supported_models"] if "/" in m["model_name"]
    ]
    models = served if args.all else args.model
    runnable = {}
    for model in models:
        try:
            runnable[model] = renderer_for(model)
        except (KeyError, ValueError) as error:
            print(
                f"skip   {model}: tinker-cookbook has no renderer for it ({error})",
                flush=True,
            )

    with ThreadPoolExecutor(args.parallel) as pool:
        rows = list(
            pool.map(lambda item: run(*item, args, args.base_url), runnable.items())
        )

    stamp = {
        "date": datetime.now(UTC).isoformat(timespec="seconds"),
        "spindle_version": info.get("spindle_version"),
        "spindle_commit": (info.get("spindle_commit") or "")[:12],
        "tinker": version("tinker"),
        "tinker_cookbook": version("tinker-cookbook"),
    }
    for row in rows:
        report(row)
    if not args.no_save:
        RESULTS.parent.mkdir(parents=True, exist_ok=True)
        with RESULTS.open("a") as handle:
            for row in rows:
                handle.write(json.dumps(stamp | row) + "\n")
    sys.exit(0 if all(row["ok"] for row in rows) else 1)


if __name__ == "__main__":
    main()
