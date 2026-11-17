"""Miles-pinned GLM model dependency with the release Bridge core preserved."""

import modal

from .image_dependencies import ignore_config_source
from .miles_image import BRIDGE_IMPORT_CACHE_INSTALL
from .miles_release_image import MILES_PATH, release_image

MILES_REPOSITORY = "https://github.com/modal-projects/miles.git"
MILES_BRANCH = "spindle-miles-v0.1.2"


def image_at(miles_commit: str) -> modal.Image:
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


# Fork commits in ancestry order, each a descendant of the one before. Append new commits; never edit an entry.
# Merged modal-projects/miles#31 on spindle-miles-v0.1.2.
image_01_464efc9 = image_at("464efc9781bbf8edf2135f0857f5aa417685eae2")
# Merged modal-projects/miles#43: per-adapter seeds through load_slot.
image_02_99e1ebe = image_at("99e1ebe8e86974e0b3c290f47f415ee47726f965")
