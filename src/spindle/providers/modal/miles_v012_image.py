"""Miles v0.1.2 release image with Spindle's Miles fork branch for v0.1.2."""

import modal

from .image_dependencies import ignore_config_source
from .miles_release_image import MILES_PATH, release_image
from .miles_v011_image import BRIDGE_IMPORT_CACHE_INSTALL

MILES_REPOSITORY = "https://github.com/modal-projects/miles.git"
MILES_BRANCH = "spindle-miles-v0.1.2"


# Pinned by commit: Modal caches this layer by command, so fetching the branch
# by name would keep serving an older head.
def image_at(miles_commit: str) -> modal.Image:
    return release_image.run_commands(
        f"git -C {MILES_PATH} fetch --depth=1 {MILES_REPOSITORY} {miles_commit}"
        f" && git -C {MILES_PATH} checkout -B {MILES_BRANCH} FETCH_HEAD",
        # Fails the build if the fork branch did not replace the release checkout.
        f"test -f {MILES_PATH}/miles_plugins/models/deepseek_v4_1/lora.py",
    ).add_local_python_source("spindle", ignore=ignore_config_source)


def glm53_image_at(miles_commit: str) -> modal.Image:
    """Miles-pinned GLM model dependency with the release Bridge core preserved."""
    return release_image.run_commands(
        f"git -C {MILES_PATH} fetch --depth=1"
        f" {MILES_REPOSITORY} {miles_commit}"
        f" && git -C {MILES_PATH} checkout --detach FETCH_HEAD",
        # Miles owns the provider and runtime pins. The installer exposes only GLM
        # model packages from the dependency, retaining the release Bridge core.
        f"pip install -r {MILES_PATH}/docker/requirements-glm53.txt",
        f"python {MILES_PATH}/docker/install_glm53_provider.py",
        BRIDGE_IMPORT_CACHE_INSTALL,
    ).add_local_python_source("spindle", ignore=ignore_config_source)


# Each series lists fork commits in ancestry order, each a descendant of the one before. Append new commits; never edit an entry.
image_01_f2f8447 = image_at("f2f8447c33ac0755a0988c30d413d334dea0a26a")
# Branch head: the GLM-5.3 provider hooks, inert without the glm53 build steps, and per-adapter seeds through load_slot.
image_02_99e1ebe = image_at("99e1ebe8e86974e0b3c290f47f415ee47726f965")

# Merged modal-projects/miles#31 on spindle-miles-v0.1.2.
glm53_image_01_464efc9 = glm53_image_at("464efc9781bbf8edf2135f0857f5aa417685eae2")
# Merged modal-projects/miles#43: per-adapter seeds through load_slot.
glm53_image_02_99e1ebe = glm53_image_at("99e1ebe8e86974e0b3c290f47f415ee47726f965")
