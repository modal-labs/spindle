"""modal run tests/manual/validate_inkling2_lora.py (temporary, non-reserved GPUs)."""

from pathlib import Path
import argparse
import shlex
import tempfile
import subprocess
import sys

import modal
from huggingface_hub import snapshot_download

from spindle.backends.deployment import backend_config
from spindle.backends.miles_arguments import defer_config_overrides
from spindle.backends.miles_config import MilesBackendConfig

from spindle.providers.modal.inkling2_image import (
    inference_image,
    trainer_runtime_image,
)

app = modal.App("spindle-inkling2-validation")
trainer_image = (
    trainer_runtime_image.pip_install("pytest==9.1.1")
    .add_local_python_source("spindle")
    .add_local_file(
        Path(__file__).parents[1] / "backends/test_inkling2_native.py",
        "/root/test_inkling2_native.py",
    )
)


@app.function(
    image=trainer_image,
    gpu="B300:8",
    cpu=8,
    memory=65536,
    timeout=1800,
    experimental_options={"exclude_reserved_capacity": True},
)
@modal.clustered(size=1)
def trainer() -> None:
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "--confcutdir=/root",
            "/root/test_inkling2_native.py",
        ],
        check=True,
    )
    subprocess.run(
        [sys.executable, "-c", "import spindle.backends.miles_runtime.actor_v020"],
        check=True,
    )
    parse_training_config()


def parse_training_config() -> None:
    # These trainer-only dependencies are absent from the sampler image.
    from spindle.configs.inkling2_small_lora_8k_b300 import Config
    from miles.utils.arguments import parse_args
    from miles.utils.external_utils.model_args_utils import load_model_args
    from spindle.backends.miles_runtime.runtime import _configure_args, _temporary_argv

    recipe = Config()
    with tempfile.TemporaryDirectory() as checkpoint:
        snapshot_download(
            recipe.model,
            revision=recipe.weights_revision,
            local_dir=checkpoint,
            allow_patterns=["*.json", "*.jinja", "*.py", "*.tiktoken"],
        )
        config = MilesBackendConfig(**backend_config(recipe, checkpoint)["miles"])
        architecture = shlex.split(load_model_args(config.model_type))
        with _temporary_argv([*architecture, *config.miles_arguments()]):

            def configure(parser: argparse.ArgumentParser) -> object:
                return defer_config_overrides(parser, config.cli_options, sys.argv)

            args = parse_args(
                add_custom_arguments=configure,
                entry="serve",
                preprocess_args=_configure_args,
            )
        assert args.megatron_to_hf_mode == "raw"
        assert args.multi_lora_n_adapters == 2
        assert not any("experts" in target for target in args.hf_lora_targets)


@app.function(
    image=inference_image,
    gpu="B300:8",
    cpu=8,
    memory=65536,
    timeout=1800,
    experimental_options={"exclude_reserved_capacity": True},
)
@modal.clustered(size=1)
def sampler() -> None:
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sglang.srt.models.inkling2; import spindle.inference.sglang",
        ],
        check=True,
    )


@app.local_entrypoint()
def main() -> None:
    trainer.remote()
    sampler.remote()
