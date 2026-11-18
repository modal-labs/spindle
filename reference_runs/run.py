"""Prepare a reference deployment or launch its Tinker workload."""

import argparse
import contextlib
import pprint
import runpy
import sys
import tomllib
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from catalog import ReferenceRun, record_run

from spindle.configuration import BaseConfig
from spindle.deployments import config_path, load


class TokenizerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: str = Field(min_length=1)
    revision: str = Field(pattern=r"^[0-9a-f]{40}$")


class ConfigOverrides(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    name: str | None = None
    trainer_gpu: str | None = None
    trainer_gpus_per_node: int | None = None
    trainer_nodes: int | None = None
    trainer_max_instances: int | None = None
    trainer_max_clients_per_instance: int | None = None
    inference_gpu: str | None = None
    inference_gpus_per_node: int | None = None
    inference_min_replicas: int | None = None
    inference_max_replicas: int | None = None
    miles_max_lora_slots: int | None = Field(
        default=None, alias="miles_cfg.max_lora_slots"
    )


class ValidationRunConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    config: str
    script: str
    tokenizer: TokenizerConfig
    config_overrides: ConfigOverrides = Field(default_factory=ConfigOverrides)
    script_args: list[str] = Field(default_factory=list)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("validation", type=Path, help="validation TOML file")
    destination = parser.add_mutually_exclusive_group(required=True)
    destination.add_argument(
        "--write-config", type=Path, help="write a new deployment Python file and exit"
    )
    destination.add_argument("--output", type=Path, help="new JSONL output file")
    destination.add_argument(
        "--record", type=Path, help="execution provenance TOML; index after success"
    )
    parser.add_argument(
        "--index", type=Path, default=Path(__file__).with_name("index.toml")
    )
    args = parser.parse_args()
    definition = ValidationRunConfig.model_validate(
        tomllib.loads(args.validation.read_text())
    )
    parent = args.validation.resolve().parent
    recipe = BaseConfig(
        **vars(load(config_path(definition.config))),
        overrides=definition.config_overrides.model_dump(
            by_alias=True, exclude_none=True
        ),
    )
    if args.write_config:
        with args.write_config.open("x") as stream:
            stream.write(
                "from spindle.configuration import BaseConfig\n\nconfig = BaseConfig(**"
                + pprint.pformat(vars(recipe), sort_dicts=False)
                + ")\n"
            )
        return
    if any(
        arg.split("=", 1)[0] in {"--base-model", "--output", "--tokenizer"}
        for arg in definition.script_args
    ):
        parser.error(
            "--base-model, --output and --tokenizer are supplied by the launcher"
        )
    entry = (
        ReferenceRun.model_validate(tomllib.loads(args.record.read_text()))
        if args.record
        else None
    )
    if entry is not None and entry.validation != args.validation.stem:
        parser.error("record validation must match the selected definition")
    registration = (
        record_run(args.index, entry)
        if entry is not None
        else contextlib.nullcontext(args.output)
    )
    with registration as output:
        launch(definition, recipe, parent, output)


def launch(
    definition: ValidationRunConfig, recipe: BaseConfig, parent: Path, output: Path
) -> None:
    script = (parent / definition.script).resolve()
    arguments = [
        str(script),
        "--base-model",
        recipe.name,
        "--output",
        str(output),
        "--tokenizer",
        definition.tokenizer.model,
        definition.tokenizer.revision,
        *definition.script_args,
    ]
    previous_argv, previous_path = sys.argv, sys.path[:]
    try:
        sys.argv = arguments
        sys.path.insert(0, str(script.parent))
        runpy.run_path(str(script), run_name="__main__")
    finally:
        sys.argv = previous_argv
        sys.path[:] = previous_path


if __name__ == "__main__":
    main()
