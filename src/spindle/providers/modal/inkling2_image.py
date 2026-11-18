"""Inkling-2-Small attention and unembedding multi-LoRA images."""

from pathlib import Path

import modal

from .image_dependencies import (
    CORE_PACKAGES,
    MEGATRON_RUNTIME_PACKAGES,
    STITCH_PACKAGE,
    TINKER_PACKAGE,
    ignore_config_source,
)
from .rollout_image import CORE_SAMPLER_PACKAGES

BASE_IMAGE = "radixark/miles:inkling-2-small@sha256:a2e3c49d5a266d9374d67ca4b706767b7bdf7d798958dd51520f61c29597e9a5"
MILES_COMMIT = "e186d97710dcc319da659f8bc25dbb674e227909"
MILES_REPOSITORY = "https://github.com/modal-projects/miles.git"

runtime_image = (
    modal.Image.from_registry(BASE_IMAGE)
    .entrypoint([])
    .env({"SGLANG_RUST_BUILD_MODE": "auto"})
)
head_lora_inference_image = (
    runtime_image.add_local_file(
        Path(__file__).with_name("inkling2-lora-head.patch"),
        "/tmp/inkling2-lora-head.patch",
        copy=True,
    )
    .run_commands(
        "cd /sgl-workspace/sglang && patch -p1 < /tmp/inkling2-lora-head.patch"
    )
    .pip_install(*CORE_SAMPLER_PACKAGES, STITCH_PACKAGE, TINKER_PACKAGE)
    .add_local_python_source("spindle", ignore=ignore_config_source)
)
trainer_runtime_image = (
    runtime_image.env(
        {
            "PYTHONPATH": "/root/Megatron-LM:$PYTHONPATH",
            "CUDA_DEVICE_MAX_CONNECTIONS": "1",
            "SPINDLE_MILES_API": "0.2",
        }
    )
    .pip_install(*CORE_PACKAGES, STITCH_PACKAGE, *MEGATRON_RUNTIME_PACKAGES)
    .run_commands(
        "rm -rf /root/miles && git init /root/miles"
        f" && git -C /root/miles fetch --depth=1 {MILES_REPOSITORY} {MILES_COMMIT}"
        f" && git -C /root/miles checkout --detach {MILES_COMMIT}",
        "pip install --no-deps --no-build-isolation -e /root/miles",
    )
)
image = trainer_runtime_image.add_local_python_source(
    "spindle", ignore=ignore_config_source
)
