"""Miles v0.2.0 with the remaining Spindle trainer fixes."""

import modal

from .image_dependencies import (
    CORE_PACKAGES,
    MEGATRON_RUNTIME_CHECK,
    MEGATRON_RUNTIME_PACKAGES,
    STITCH_PACKAGE,
    ignore_config_source,
)

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


# Fork commits in ancestry order, each a descendant of the one before. Append new commits; never edit an entry.
# Merged modal-projects/miles#44 after Beam LoRA support (#37): per-adapter seeds through load_slot.
image_02_8c5de26 = image_at("8c5de2650a6876c2ca17ff221c7e0ba8a2d7b84a")

# Native canonical and routed-expert adapters, including Nemotron Nano (#47, #49).
image_03_e814245 = image_at("e8142459617f8220825ddbacf054b8b17cabdd82")

# GLM output heads, Omni layouts and expert routing (#50), plus native GLM 256K memory bounds (#52).
image_04_26989c0 = image_at("26989c0bac5cc7e1a9b62a7c8847ab126fad5e22")
