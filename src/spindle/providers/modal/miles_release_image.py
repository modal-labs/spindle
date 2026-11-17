"""Miles v0.1.2 release image, the first release that trains DeepSeek-V4.1."""

import subprocess

from .image_dependencies import (
    MEGATRON_RUNTIME_CHECK,
    MEGATRON_RUNTIME_PACKAGES,
)
from .miles_image_common import MILES_EXTRA_PACKAGES, miles_base_image

MILES_RELEASE = "v0.1.2"
BASE_IMAGE = (
    f"radixark/miles:{MILES_RELEASE}"
    "@sha256:c81d8ba7b91571aac8e0bb2be372c6a9664e1ff7b88f518a8891861bae83e3b7"
)
MILES_COMMIT = "a22fd824191c64ba7c5d6c531f125406e4ce9cc6"
MILES_PATH = "/root/miles"
# Megatron-LM is only partially installed as a package; some model
# construction imports resolve from the source tree.
MEGATRON_PATH = "/root/Megatron-LM"


def _check_release(miles_commit: str) -> None:
    head = subprocess.check_output(
        ["git", "-C", MILES_PATH, "rev-parse", "HEAD"], text=True
    ).strip()
    assert head == miles_commit, f"Miles is at {head}, not {miles_commit}"


release_image = (
    miles_base_image(
        BASE_IMAGE,
        {
            "LD_LIBRARY_PATH": "/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH",
            "PYTHONPATH": f"{MEGATRON_PATH}:$PYTHONPATH",
            "CUDA_DEVICE_MAX_CONNECTIONS": "1",
        },
    )
    .pip_install(
        *MEGATRON_RUNTIME_PACKAGES,
        *MILES_EXTRA_PACKAGES,
    )
    .run_function(_check_release, kwargs={"miles_commit": MILES_COMMIT})
    .run_commands(MEGATRON_RUNTIME_CHECK)
)
