"""Inkling-2-Small attention and unembedding multi-LoRA images."""

import modal

from .image_dependencies import (
    CORE_PACKAGES,
    MEGATRON_RUNTIME_PACKAGES,
    STITCH_PACKAGE,
    TINKER_PACKAGE,
    ignore_config_source,
)
from .inkling2_patch import INKLING2_ATTENTION_PATCH
from .rollout_image import CORE_SAMPLER_PACKAGES

BASE_IMAGE = "radixark/miles:inkling-2-small@sha256:a2e3c49d5a266d9374d67ca4b706767b7bdf7d798958dd51520f61c29597e9a5"
MILES_COMMIT = "384d73b13da2295dc9bb094da177bb5f4c41b682"
MILES_REPOSITORY = "https://github.com/modal-projects/miles.git"
INKLING2_BASE_COMMIT = "a18187f3efe49f6d1a8d2fbf1ffab2061e5ac69b"
INKLING2_COMMIT = "eac7f8cc79c09e7b3dcbb6bef3029dc8323390f0"

runtime_image = (
    modal.Image.from_registry(BASE_IMAGE)
    .entrypoint([])
    .env({"SGLANG_RUST_BUILD_MODE": "auto"})
)
inference_image = runtime_image.pip_install(
    *CORE_SAMPLER_PACKAGES, STITCH_PACKAGE, TINKER_PACKAGE
).add_local_python_source("spindle", ignore=ignore_config_source)
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
        "git -C /root/miles fetch --depth=1 https://github.com/radixark/miles.git"
        f" {INKLING2_BASE_COMMIT} {INKLING2_COMMIT}"
        f" && git -C /root/miles diff {INKLING2_BASE_COMMIT} {INKLING2_COMMIT}"
        " > /tmp/inkling2.patch"
        " && git -C /root/miles apply --check /tmp/inkling2.patch"
        " && git -C /root/miles apply /tmp/inkling2.patch",
        "git -C /root/miles apply --check - <<'PATCH'\n"
        + INKLING2_ATTENTION_PATCH
        + "PATCH\n",
        "git -C /root/miles apply - <<'PATCH'\n" + INKLING2_ATTENTION_PATCH + "PATCH\n",
        "pip install --no-deps --no-build-isolation -e /root/miles",
    )
)
image = trainer_runtime_image.add_local_python_source(
    "spindle", ignore=ignore_config_source
)
