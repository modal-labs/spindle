"""Plan and execute trainer curves using the existing revision benchmark runner."""

import argparse
import json
import subprocess
import sys
import threading
import uuid
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from spindle.configuration import BaseConfig
from spindle.deployments import DeploymentConfig, load

from .schema import BenchmarkConfig

ROOT = Path(__file__).resolve().parents[3]
CONFIGS = Path(__file__).resolve().parents[1] / "configs"
RECIPE_LOAD_LOCK = threading.Lock()


def load_recipe(path):
    with RECIPE_LOAD_LOCK:
        return load(path)


class TrainingSweep(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = "trainer-scaling"
    deployment: str
    suite: Literal["eval", "trainer", "weight-sync"] = "trainer"
    batch_sizes: list[int] = Field(
        default_factory=lambda: [1, 2, 4, 8], min_length=1, max_length=16
    )
    context_lengths: list[int] = Field(
        default_factory=lambda: [512, 2048, 8192], min_length=1, max_length=16
    )
    clients: list[int] = Field(default_factory=lambda: [1], min_length=1, max_length=8)
    steps: int = Field(default=10, ge=1, le=10000)
    warmup_steps: int = Field(default=2, ge=0)
    rank: int = Field(default=8, ge=1)
    seed: int = 42
    learning_rate: float = Field(default=1e-5, gt=0)
    timeout_s: float = Field(default=7200, gt=0)
    deployment_overrides: dict = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_axes(self):
        for name in ("batch_sizes", "context_lengths", "clients"):
            values = getattr(self, name)
            if any(
                value < (2 if name == "context_lengths" else 1) for value in values
            ) or len(set(values)) != len(values):
                raise ValueError(f"{name} must contain unique positive values")
        return self

    def workloads(self, root=ROOT):
        recipe_path = (root / self.deployment).resolve()
        if not recipe_path.is_relative_to(root.resolve()):
            raise ValueError("deployment must be a repository-relative Python config")
        if (
            not recipe_path.is_file()
            and self.deployment == "src/spindle/configs/" + recipe_path.name
        ):
            recipe_path = CONFIGS / recipe_path.name
        recipe = load_recipe(recipe_path)
        try:
            recipe = DeploymentConfig.create(
                BaseConfig(**vars(recipe), overrides=self.deployment_overrides)
            ).recipe
        except (AttributeError, TypeError) as exc:
            raise ValueError("Invalid deployment override type or nested path") from exc
        context_limit = recipe.max_context_length
        if max(self.context_lengths) > context_limit:
            raise ValueError(f"context length exceeds recipe limit {context_limit}")
        client_limit = recipe.trainer_max_clients_per_instance
        if max(self.clients) > client_limit:
            raise ValueError(
                f"clients exceed recipe capacity {client_limit}; set an explicit deployment override"
            )
        settings = self.model_dump(exclude={"batch_sizes"})
        return [
            BenchmarkConfig.model_validate(
                {
                    **settings,
                    "name": f"{self.name}-batch-{batch}",
                    "experiment": self.name,
                    "model": recipe.model,
                    "parameterization": recipe.parameterization,
                    "batch_size": batch,
                }
            )
            for batch in self.batch_sizes
        ]


def bundled_recipes():
    rows = []
    for path in sorted(CONFIGS.glob("*.py")):
        if path.name.startswith("_"):
            continue
        recipe = load_recipe(path)
        rows.append(
            {
                "path": "src/spindle/configs/" + path.name,
                "name": recipe.name,
                "model": recipe.model,
                "parameterization": recipe.parameterization,
                "backend": recipe.backend,
                "max_context_length": recipe.max_context_length,
                "max_clients": recipe.trainer_max_clients_per_instance,
                "gpu": recipe.trainer_gpu,
                "gpus_per_node": recipe.trainer_gpus_per_node,
                "nodes": recipe.trainer_nodes,
            }
        )
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path)
    parser.add_argument("--environment", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--collector-app", default="frontier-training-dev")
    parser.add_argument("--directory", type=Path, default=Path("benchmark-results"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    plan = TrainingSweep.model_validate_json(args.plan.read_text())
    workloads = plan.workloads()
    if args.dry_run:
        print(json.dumps([config.model_dump() for config in workloads], indent=2))
        return
    target = args.directory.resolve() / "plans" / uuid.uuid4().hex[:12]
    target.mkdir(parents=True)
    (target / "plan.json").write_text(plan.model_dump_json(indent=2))
    for index, config in enumerate(workloads):
        path = target / f"point-{index}.json"
        path.write_text(config.model_dump_json(indent=2))
        subprocess.run(
            [
                sys.executable,
                "-m",
                "modal",
                "run",
                "-e",
                args.environment,
                "-m",
                "spindle.benchmarks.controller",
                "--environment",
                args.environment,
                "--config",
                str(path),
                "--collector-app",
                args.collector_app,
                "--revision",
                args.revision,
                "--directory",
                str(args.directory.resolve()),
            ],
            cwd=ROOT,
            check=True,
        )


if __name__ == "__main__":
    main()
