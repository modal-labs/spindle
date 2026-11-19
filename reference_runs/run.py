"""Prepare a reference deployment or launch its Tinker workload."""

import argparse
import importlib.metadata
import os
import pprint
import re
import runpy
import sys
import tomllib
from pathlib import Path
from typing import cast

import tinker
from tinker._constants import RAW_RESPONSE_HEADER
from tinker.lib.client_connection_pool_type import ClientConnectionPoolType
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
    sglang_tp_size: int | None = Field(default=None, alias="sglang_cfg.tp_size")
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
    root = Path(__file__).resolve().parent
    indexed = args.output.resolve().is_relative_to(root / "outputs")
    validation = args.validation.resolve()
    if indexed and (
        validation.parent != root / "configs" or validation.suffix != ".toml"
    ):
        parser.error(
            "Indexed outputs require a TOML definition in reference_runs/configs/; "
            "use a scratch --output for other definitions"
        )
    if any(
        arg.split("=", 1)[0] in {"--base-model", "--output", "--tokenizer"}
        for arg in definition.script_args
    ):
        parser.error(
            "--base-model, --output and --tokenizer are supplied by the launcher"
        )
    commit = (
        re.search(r"-([0-9a-f]{7,40})(?:-|$)", recipe.trainer_image)
        if recipe.trainer_image.startswith("miles-")
        else None
    )
    entry = ReferenceRun(
        id=args.output.stem,
        validation=validation.stem,
        output=os.path.relpath(args.output.resolve(), args.index.resolve().parent),
        trainer_image=recipe.trainer_image,
        miles_commit=commit.group(1) if commit else "unknown",
        tinker_sdk=importlib.metadata.version("tinker"),
        hardware=(
            f"Configured trainers: {recipe.trainer_nodes} x "
            f"{recipe.trainer_gpus_per_node} {recipe.trainer_gpu}; samplers: "
            f"{recipe.inference_min_replicas}-{recipe.inference_max_replicas} x "
            f"{recipe.inference_gpus_per_node} {recipe.inference_gpu}."
        ),
        region=recipe.platform["modal"].get("region"),
        notes=(
            "Server metadata from get_server_capabilities; resolved recipe settings; "
            "remote runtime and hardware are not attested. "
            f"Configured sampler image: {recipe.inference_image}."
        ),
    )
    with record_run(args.index, entry, indexed=indexed) as output:
        if indexed:
            metadata = server_metadata()
            entry.spindle_commit = metadata["spindle_commit"]
            entry.spindle_version = metadata["spindle_version"]
        launch(definition, recipe, parent, output)


def server_metadata() -> dict[str, str | None]:
    service = tinker.ServiceClient()
    holder = service._get_rest_holder()

    async def read() -> object:
        with holder.aclient(ClientConnectionPoolType.TRAIN) as client:
            # Typed capability responses discard unknown server metadata.
            raw_client = client.with_options(
                default_headers={RAW_RESPONSE_HEADER: "raw"}
            )
            response = cast(
                tinker.AsyncAPIResponse[tinker.types.GetServerCapabilitiesResponse],
                await raw_client.service.get_server_capabilities(),
            )
            return await response.json()

    values = holder.run_coroutine_threadsafe(holder.execute_with_retries(read)).result()
    if not isinstance(values, dict):
        raise ValueError("Server capabilities must be a JSON object")
    return {
        key: value if isinstance(value := values.get(key), str) and value else None
        for key in ("spindle_commit", "spindle_version")
    }


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
    except SystemExit as exc:
        if exc.code not in (None, 0):
            raise
    finally:
        sys.argv = previous_argv
        sys.path[:] = previous_path


if __name__ == "__main__":
    main()
