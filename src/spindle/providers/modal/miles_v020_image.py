"""Miles v0.2.0 with the remaining Spindle trainer fixes."""

import modal

from .image_dependencies import (
    CORE_PACKAGES,
    MEGATRON_RUNTIME_CHECK,
    MEGATRON_RUNTIME_PACKAGES,
    STITCH_PACKAGE,
    ignore_config_source,
)
from .miles_image_common import miles_base_image

MILES_RELEASE_COMMIT = "8e419cc739fb09b5f21085aeacc7f6fd951084e1"
MILES_REPOSITORY = "https://github.com/modal-projects/miles.git"
MILES_BRANCH = "spindle-miles-v0.2.0"
BASE_IMAGE = "radixark/miles:v0.2.0@sha256:c5e9ebe1dbb02faa094cf5036b57c4f3aadac9a53992f4d143823a6c20fb377c"


# Editing this rebuilds every image below; add a new builder instead.
def image_at(miles_commit: str) -> modal.Image:
    return (
        modal.Image.from_registry(BASE_IMAGE)
        .entrypoint([])
        .env(
            {
                "LD_LIBRARY_PATH": "/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH",
                "PYTHONPATH": "/root/Megatron-LM:$PYTHONPATH",
                "CUDA_DEVICE_MAX_CONNECTIONS": "1",
                "SPINDLE_MILES_API": "0.2",
            }
        )
        .pip_install(*CORE_PACKAGES, STITCH_PACKAGE, *MEGATRON_RUNTIME_PACKAGES)
        .run_commands(
            f'test "$(git -C /root/miles rev-parse HEAD)" = "{MILES_RELEASE_COMMIT}"',
            f"git -C /root/miles fetch --depth=1 {MILES_REPOSITORY} {miles_commit}"
            f" && git -C /root/miles checkout -B {MILES_BRANCH} FETCH_HEAD",
            MEGATRON_RUNTIME_CHECK,
        )
        .add_local_python_source("spindle", ignore=ignore_config_source)
    )


def grouped_gemm_image_at(miles_commit: str) -> modal.Image:
    """Enable TE's existing fused grouped backend and its required cuBLAS runtime."""
    library = "/opt/cublas-13.3/nvidia/cu13/lib"
    return (
        miles_base_image(
            BASE_IMAGE,
            {
                "LD_LIBRARY_PATH": "/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH",
                "PYTHONPATH": "/root/Megatron-LM:$PYTHONPATH",
                "CUDA_DEVICE_MAX_CONNECTIONS": "1",
                "SPINDLE_MILES_API": "0.2",
            },
        )
        .pip_install(*MEGATRON_RUNTIME_PACKAGES)
        .run_commands(
            f'test "$(git -C /root/miles rev-parse HEAD)" = "{MILES_RELEASE_COMMIT}"',
            f"git -C /root/miles fetch --depth=1 {MILES_REPOSITORY} {miles_commit}"
            f" && git -C /root/miles checkout -B {MILES_BRANCH} FETCH_HEAD",
            MEGATRON_RUNTIME_CHECK,
            "uv pip install --python /opt/sglang/bin/python --no-deps"
            " --target /opt/cublas-13.3 nvidia-cublas==13.3.0.5",
        )
        .env(
            {
                "NVTE_GROUPED_LINEAR_USE_FUSED_GROUPED_GEMM": "1",
                "LD_PRELOAD": f"{library}/libcublasLt.so.13:{library}/libcublas.so.13",
            }
        )
        .add_local_python_source("spindle", ignore=ignore_config_source)
    )


# Fork commits in ancestry order, each a descendant of the one before. Append new commits; never edit an entry.
# Merged modal-projects/miles#44 after Beam LoRA support (#37): per-adapter seeds through load_slot.
image_02_8c5de26 = image_at("8c5de2650a6876c2ca17ff221c7e0ba8a2d7b84a")

# Native canonical and routed-expert adapters, including Nemotron Nano (#47, #49).
image_03_e814245 = image_at("e8142459617f8220825ddbacf054b8b17cabdd82")


# Native shared-prefix execution and bounded-memory native LoRA (#53).
image_04_7da41f2 = grouped_gemm_image_at("7da41f2f7d024fdb60d4ce456fb04af3cbd1ff73")
