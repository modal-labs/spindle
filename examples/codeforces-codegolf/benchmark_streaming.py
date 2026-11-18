"""Compare complete codegolf updates with and without streamed minibatches."""

from __future__ import annotations

import asyncio
import dataclasses
import hashlib
import json
import logging
import os
import statistics
from pathlib import Path

import modal
import spindle
from spindle.engines import qwen3_5_9b_full_64k
from spindle.providers.modal.scoped import control_image

from codegolf.config import config_for
from codegolf.train import train

APP_NAME = "spindle-codegolf-streaming-benchmark"
VOLUME_NAME = APP_NAME
PROXY_SECRET = os.environ.get("CODEGOLF_PROXY_SECRET", "spindle-proxy")
app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True, version=2)
image = control_image("transformers==5.16.1", "jinja2==3.1.6").add_local_python_source(
    "codegolf"
)


@app.function(
    image=image,
    volumes={"/runs": volume},
    secrets=[modal.Secret.from_name(PROXY_SECRET)],
    nonpreemptible=True,
    timeout=7200,
    memory=8192,
)
def benchmark(
    label: str,
    commit: str,
    proxy_secret: str,
    steps: int = 6,
    warmup: int = 2,
    max_tokens: int = 2048,
):
    if not label or Path(label).name != label or label in {".", ".."}:
        raise ValueError("label must be a single directory name")
    if not 0 <= warmup < steps:
        raise ValueError("require 0 <= warmup < steps")
    logging.basicConfig(level=logging.INFO)
    root = Path("/runs") / label
    root.mkdir(exist_ok=False)
    engine = qwen3_5_9b_full_64k()
    report = {
        "commit": commit,
        "engine": engine.name,
        "trainer_gpu": engine.trainer_gpu,
        "sampler_gpu": engine.sampler_gpu,
        "sampler_replicas": 4,
        "max_tokens": max_tokens,
        "steps": steps,
        "warmup": warmup,
        "dataset_sha256": hashlib.sha256(
            Path("/runs/problems.json").read_bytes()
        ).hexdigest(),
        "runs": [],
    }
    with spindle.run(
        engine=engine,
        name=APP_NAME,
        proxy_secret=modal.Secret.from_name(proxy_secret),
        requires_proxy_auth=True,
        checkpoint_volume=APP_NAME + "-checkpoints",
        latest=spindle.Pool(min_containers=4, max_containers=4, scaledown_window=1200),
    ) as (url, api_key):
        os.environ["TINKER_BASE_URL"] = url
        os.environ["TINKER_API_KEY"] = api_key
        for index, variant in enumerate(
            ("batch-v11", "stream-v11", "stream-v11", "batch-v11")
        ):
            run_root = root / f"{index}-{variant}"
            run_root.mkdir()
            (run_root / "validated.json").write_bytes(
                Path("/runs/validated.json").read_bytes()
            )
            cfg = dataclasses.replace(
                config_for(variant, steps=steps),
                max_tokens=max_tokens,
                eval_problems=2,
                eval_every=steps + 1,
                checkpoint_every=steps + 1,
                rollout_min_replicas=4,
                rollout_max_replicas=4,
            )
            asyncio.run(
                train(
                    run_root, Path("/runs/problems.json"), app, cfg, volume.commit.aio
                )
            )
            metrics = [
                json.loads(path.read_text())
                for path in sorted((run_root / "metrics").glob("*.json"))
            ]
            measured = [row for row in metrics if row["step"] > warmup]
            summary = {
                "variant": variant,
                "config": dataclasses.asdict(cfg),
                "run_dir": run_root.name,
                "median_update_seconds": statistics.median(
                    row["seconds"] + row["publication_seconds"] for row in measured
                ),
                "mean_completion_tokens": statistics.fmean(
                    row["completion_tokens"] for row in measured
                ),
                "mean_reward": statistics.fmean(row["reward"] for row in measured),
                "metrics": measured,
                "recovery_events": len(list((run_root / "attempts").glob("*.json"))),
            }
            report["runs"].append(summary)
            (root / "result.json").write_text(json.dumps(report, indent=2))
            volume.commit()
            print(
                "BENCHMARK_RUN "
                + json.dumps(
                    {k: v for k, v in summary.items() if k not in {"metrics", "config"}}
                ),
                flush=True,
            )
    report["scope_closed"] = True
    (root / "result.json").write_text(json.dumps(report, indent=2))
    volume.commit()
    return json.dumps(report)
