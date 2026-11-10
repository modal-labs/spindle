"""Beam training and sampling runtimes with separate Torch-compatible pins."""

from .image_dependencies import CORE_PACKAGES, ignore_config_source
from .miles_image import BRIDGE_IMPORT_CACHE_PATCH
from .miles_release_image import MILES_PATH, release_image
from .rollout_image import runtime_image_for

MILES_COMMIT = "a1a4ae31dbbd01a00fc6396a5a8b9fb954101505"
TRAINER_SGLANG_COMMIT = "9ac1c65bec3dc832766d03970e545da5bd6fcb6f"
SAMPLER_SGLANG_COMMIT = "53a6ab040738f95eaf74bfc7378b9235ec7b576e"

trainer_image = release_image.run_commands(
    f"git -C {MILES_PATH} fetch --depth=1"
    f" https://github.com/modal-projects/miles.git {MILES_COMMIT}"
    f" && git -C {MILES_PATH} checkout --detach FETCH_HEAD",
    # Beam's config must match the Torch 2.13 trainer and its compiled extensions.
    "git clone --filter=blob:none --single-branch --branch beam-miles"
    " https://github.com/sgl-project/sglang.git /tmp/beam-sglang"
    f" && git -C /tmp/beam-sglang checkout {TRAINER_SGLANG_COMMIT}"
    " && cp -a /tmp/beam-sglang/python/. /sgl-workspace/sglang/python/"
    " && rm -rf /tmp/beam-sglang",
    "python - <<'PY'\n"
    "from importlib.metadata import distribution\n"
    "import subprocess\n"
    "root = distribution('megatron-bridge').locate_file('')\n"
    "subprocess.run(['patch', '-p1', '-d', str(root)], input="
    + repr(BRIDGE_IMPORT_CACHE_PATCH)
    + ", text=True, check=True)\nPY\n",
).add_local_python_source("spindle", ignore=ignore_config_source)

inference_image = runtime_image_for(
    image="lmsysorg/sglang:v0.5.23",
    branch="fred/beam-spindle",
    revision=SAMPLER_SGLANG_COMMIT,
    core_packages=tuple(p.replace("==1.43.0", "==1.45.2") for p in CORE_PACKAGES),
).add_local_python_source("spindle", ignore=ignore_config_source)
