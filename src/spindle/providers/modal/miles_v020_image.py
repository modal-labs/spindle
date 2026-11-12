"""Miles v0.2.0 trainer image with its released SGLang and compiled dependencies."""

import subprocess

import modal

from .image_dependencies import (
    CORE_PACKAGES,
    MEGATRON_RUNTIME_CHECK,
    MEGATRON_RUNTIME_PACKAGES,
    STITCH_PACKAGE,
    TINKER_PACKAGE,
    ignore_config_source,
)

MILES_RELEASE = "v0.2.0"
BASE_IMAGE = (
    f"radixark/miles:{MILES_RELEASE}"
    "@sha256:c5e9ebe1dbb02faa094cf5036b57c4f3aadac9a53992f4d143823a6c20fb377c"
)
MILES_RELEASE_COMMIT = "8e419cc739fb09b5f21085aeacc7f6fd951084e1"
MILES_BRANCH = "fred/beam-spindle-v020"
MILES_COMMIT = "69abc2c0f28ab2e86c6dffbe8cd705a5d6e9a1ad"
MILES_PATH = "/root/miles"
# Megatron-LM is only partially installed as a package; some model
# construction imports resolve from the source tree.
MEGATRON_PATH = "/root/Megatron-LM"


def _check_release(miles_commit: str) -> None:
    head = subprocess.check_output(
        ["git", "-C", MILES_PATH, "rev-parse", "HEAD"], text=True
    ).strip()
    assert head == miles_commit, f"Miles is at {head}, not {miles_commit}"


image = (
    modal.Image.from_registry(BASE_IMAGE)
    .entrypoint([])
    .env(
        {
            "LD_LIBRARY_PATH": "/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH",
            "PYTHONPATH": f"{MEGATRON_PATH}:$PYTHONPATH",
            "CUDA_DEVICE_MAX_CONNECTIONS": "1",
        }
    )
    .pip_install(*CORE_PACKAGES, STITCH_PACKAGE)
    .pip_install(
        *(p for p in MEGATRON_RUNTIME_PACKAGES if p != TINKER_PACKAGE),
        "tinker==0.36.0",
        "pyyaml>=6.0.2",
        "opentelemetry-exporter-otlp==1.43.0",
        "opentelemetry-exporter-otlp-proto-grpc==1.43.0",
        "opentelemetry-exporter-otlp-proto-http==1.43.0",
    )
    .run_function(_check_release, kwargs={"miles_commit": MILES_RELEASE_COMMIT})
    .run_commands(
        f"git -C {MILES_PATH} fetch --depth=1"
        f" https://github.com/modal-projects/miles.git {MILES_COMMIT}"
        f" && git -C {MILES_PATH} checkout -B {MILES_BRANCH} FETCH_HEAD",
    )
    .run_commands(MEGATRON_RUNTIME_CHECK)
    .add_local_python_source("spindle", ignore=ignore_config_source)
)
