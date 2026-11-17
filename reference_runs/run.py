"""Prepare a reference deployment or launch its Tinker workload."""

import argparse
import pprint
import runpy
import sys
import tomllib
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from spindle.configuration import BaseConfig
from spindle.deployments import config_path, load


class TokenizerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: str = Field(min_length=1)
    revision: str = Field(pattern=r"^[0-9a-f]{40}$")


class ValidationRunConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    config: str
    script: str
    tokenizer: TokenizerConfig
    config_overrides: dict[str, object] = Field(default_factory=dict)
    script_args: list[str] = Field(default_factory=list)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("validation", type=Path, help="validation TOML file")
    destination = parser.add_mutually_exclusive_group(required=True)
    destination.add_argument(
        "--write-config", type=Path, help="write a new deployment Python file and exit"
    )
    destination.add_argument("--output", type=Path, help="new JSONL output file")
    args, extra = parser.parse_known_args()
    if extra[:1] == ["--"]:
        extra = extra[1:]
    definition = ValidationRunConfig.model_validate(
        tomllib.loads(args.validation.read_text())
    )
    parent = args.validation.resolve().parent
    recipe = BaseConfig(
        **vars(load(config_path(definition.config))),
        overrides=definition.config_overrides,
    )
    if args.write_config:
        if extra:
            parser.error("script arguments cannot be used with --write-config")
        with args.write_config.open("x") as stream:
            stream.write(
                "from spindle.configuration import BaseConfig\n\nconfig = BaseConfig(**"
                + pprint.pformat(vars(recipe), sort_dicts=False)
                + ")\n"
            )
        return
    script_args = [*definition.script_args, *extra]
    if any(
        arg.split("=", 1)[0] in {"--base-model", "--output", "--tokenizer"}
        for arg in script_args
    ):
        parser.error(
            "--base-model, --output and --tokenizer are supplied by the launcher"
        )
    script = (parent / definition.script).resolve()
    arguments = [
        str(script),
        "--base-model",
        recipe.name,
        "--output",
        str(args.output),
        "--tokenizer",
        definition.tokenizer.model,
        definition.tokenizer.revision,
        *script_args,
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
