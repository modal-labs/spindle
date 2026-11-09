"""Miles-pinned GLM model dependency with the release Bridge core preserved."""

from .bridge_import_cache import BRIDGE_IMPORT_CACHE_PATCH
from .image_dependencies import ignore_config_source
from .miles_release_image import MILES_PATH, release_image

# Merged modal-projects/miles#31 on spindle-miles-v0.1.2.
MILES_COMMIT = "464efc9781bbf8edf2135f0857f5aa417685eae2"

image = release_image.run_commands(
    f"git -C {MILES_PATH} fetch --depth=1"
    f" https://github.com/modal-projects/miles.git {MILES_COMMIT}"
    f" && git -C {MILES_PATH} checkout --detach FETCH_HEAD",
    # Miles owns the provider and runtime pins. The installer exposes only GLM
    # model packages from the dependency, retaining the release Bridge core.
    f"pip install -r {MILES_PATH}/docker/requirements-glm53.txt",
    f"python {MILES_PATH}/docker/install_glm53_provider.py",
    "python - <<'PY'\n"
    "from importlib.metadata import distribution\n"
    "import subprocess\n"
    "root = distribution('megatron-bridge').locate_file('')\n"
    "subprocess.run(['patch', '-p1', '-d', str(root)], input="
    + repr(BRIDGE_IMPORT_CACHE_PATCH)
    + ", text=True, check=True)\nPY\n",
).add_local_python_source("spindle", ignore=ignore_config_source)
