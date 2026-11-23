"""DeepSeek-V4.1-Flash samplers with upstream's DSPARK draft metadata fix (sgl-project/sglang#46891)."""

from .image_dependencies import ignore_config_source
from .rollout_image import (
    CORE_SAMPLER_PACKAGES,
    SGLANG_BRANCH,
    SGLANG_IMAGE,
    runtime_image_for,
)

# Fork commits in ancestry order, each a descendant of the one before. Append new commits; never edit an entry.
image_01_5b3b8c8 = runtime_image_for(
    image=SGLANG_IMAGE,
    branch=SGLANG_BRANCH,
    revision="5b3b8c80910ff58128fce09c93977689f51c79e7",
    core_packages=CORE_SAMPLER_PACKAGES,
).add_local_python_source("spindle", ignore=ignore_config_source)
