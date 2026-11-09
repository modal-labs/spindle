"""GLM-only Bridge dependency, pinned by the Miles fork without model vendoring."""

from .bridge_import_cache import BRIDGE_IMPORT_CACHE_PATCH
from .image_dependencies import ignore_config_source
from .miles_release_image import MILES_PATH, release_image

MILES_COMMIT = "b86f00f3f21f2dbd7bfcc8beefe32d374d4eac35"

# This Bridge branch has the same grouped-import loop as the release, with an
# older import header. Keep the existing memory bound while it is pinned.
IMPORT_CACHE_PATCH = BRIDGE_IMPORT_CACHE_PATCH.replace(
    " import math\n"
    " import re\n"
    " import warnings\n"
    "+from collections import Counter\n"
    " from dataclasses import dataclass, field, fields, is_dataclass\n"
    " from pathlib import Path\n"
    " from typing import (",
    " import math\n"
    " import re\n"
    "+from collections import Counter\n"
    " from dataclasses import dataclass\n"
    " from typing import (\n"
    "     Any,\n"
    "     Callable,",
)

image = release_image.run_commands(
    f"git -C {MILES_PATH} fetch --depth=1"
    f" https://github.com/modal-projects/miles.git {MILES_COMMIT}"
    f" && git -C {MILES_PATH} checkout --detach FETCH_HEAD",
    # Miles owns the Bridge/Transformers/TileLang pins. The GLM branch is
    # older than the shared Bridge, so install it only for these recipes.
    "pip install --no-deps --no-build-isolation"
    f" -r {MILES_PATH}/docker/requirements-glm53.txt",
    "python - <<'PY'\n"
    "from importlib.metadata import distribution\n"
    "import subprocess\n"
    "root = distribution('megatron-bridge').locate_file('')\n"
    "subprocess.run(['patch', '-p1', '-d', str(root)], input="
    + repr(IMPORT_CACHE_PATCH)
    + ", text=True, check=True)\nPY\n",
).add_local_python_source("spindle", ignore=ignore_config_source)
