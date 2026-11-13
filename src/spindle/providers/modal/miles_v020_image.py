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
MILES_BRANCH = "spindle-v020-regression-tests"
MILES_COMMIT = "a94f7ae2ab58e6f05ba6747a48a8c43f6f4ef855"
BASE_IMAGE = "radixark/miles:v0.2.0@sha256:c5e9ebe1dbb02faa094cf5036b57c4f3aadac9a53992f4d143823a6c20fb377c"

image = (
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
        f"git -C /root/miles fetch --depth=1 {MILES_REPOSITORY} {MILES_COMMIT}"
        f" && git -C /root/miles checkout -B {MILES_BRANCH} FETCH_HEAD",
        MEGATRON_RUNTIME_CHECK,
    )
    .add_local_python_source("spindle", ignore=ignore_config_source)
)
