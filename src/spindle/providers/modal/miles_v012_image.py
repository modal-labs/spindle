"""Miles v0.1.2 release image with Spindle's Miles fork branch for v0.1.2."""

from .image_dependencies import ignore_config_source
from .miles_release_image import MILES_PATH, release_image
from .miles_image import BRIDGE_IMPORT_CACHE_PATCH

MILES_REPOSITORY = "https://github.com/modal-projects/miles.git"
MILES_BRANCH = "fred/beam-spindle"
# Head of MILES_BRANCH. Modal caches this layer by command, so fetching the branch
# by name would keep serving an older head.
MILES_COMMIT = "6f5a4a8429e4f57a65a7251facb7d3d9faf0b3ff"
TRAINER_SGLANG_COMMIT = "9ac1c65bec3dc832766d03970e545da5bd6fcb6f"

image = release_image.run_commands(
    f"git -C {MILES_PATH} fetch --depth=1 {MILES_REPOSITORY} {MILES_COMMIT}"
    f" && git -C {MILES_PATH} checkout -B {MILES_BRANCH} FETCH_HEAD",
    # Keep SGLang's model configs compatible with the trainer's Torch 2.13 extensions.
    "git -C /sgl-workspace/sglang fetch --depth=1"
    f" https://github.com/sgl-project/sglang.git {TRAINER_SGLANG_COMMIT}"
    " && git -C /sgl-workspace/sglang checkout --detach -f FETCH_HEAD"
    " && SGLANG_BUILD_RUST_EXTS=none pip install -e"
    " '/sgl-workspace/sglang/python[all]' --no-deps --no-build-isolation",
    # Fails the build if the fork branch did not replace the release checkout.
    f"test -f {MILES_PATH}/miles_plugins/models/deepseek_v4_1/lora.py",
    "python - <<'PY'\n"
    "from importlib.metadata import distribution\n"
    "import subprocess\n"
    "root = distribution('megatron-bridge').locate_file('')\n"
    "subprocess.run(['patch', '-p1', '-d', str(root)], input="
    + repr(BRIDGE_IMPORT_CACHE_PATCH)
    + ", text=True, check=True)\nPY\n",
).add_local_python_source("spindle", ignore=ignore_config_source)
