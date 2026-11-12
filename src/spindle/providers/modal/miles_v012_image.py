"""Miles v0.1.2 release image with Spindle's Miles fork branch for v0.1.2."""

from .image_dependencies import ignore_config_source
from .miles_release_image import MILES_PATH, release_image
from .miles_image import BRIDGE_IMPORT_CACHE_PATCH, patch_bridge_import_cache

MILES_REPOSITORY = "https://github.com/modal-projects/miles.git"
# Modal caches this layer by command; pin the fork commit rather than a branch.
MILES_COMMIT = "6f5a4a8429e4f57a65a7251facb7d3d9faf0b3ff"
TRAINER_SGLANG_COMMIT = "9ac1c65bec3dc832766d03970e545da5bd6fcb6f"

image = (
    release_image.run_commands(
        f"git -C {MILES_PATH} fetch --depth=1 {MILES_REPOSITORY} {MILES_COMMIT}"
        f" && git -C {MILES_PATH} checkout --detach FETCH_HEAD",
        # Keep SGLang's model configs compatible with the trainer's Torch 2.13 extensions.
        "git -C /sgl-workspace/sglang fetch --depth=1"
        f" https://github.com/sgl-project/sglang.git {TRAINER_SGLANG_COMMIT}"
        " && git -C /sgl-workspace/sglang checkout --detach -f FETCH_HEAD"
        " && SGLANG_BUILD_RUST_EXTS=none pip install -e"
        " '/sgl-workspace/sglang/python[all]' --no-deps --no-build-isolation",
        # Fails the build if the fork branch did not replace the release checkout.
        f"test -f {MILES_PATH}/miles_plugins/models/deepseek_v4_1/lora.py",
    )
    .run_function(
        patch_bridge_import_cache, kwargs={"patch": BRIDGE_IMPORT_CACHE_PATCH}
    )
    .add_local_python_source("spindle", ignore=ignore_config_source)
)
