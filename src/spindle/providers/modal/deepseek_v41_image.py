"""DeepSeek-V4.1-Flash samplers stay on SGLang v0.5.21.

On v0.5.23, a Flash LoRA sampler hits a GPU memory fault on its first batch
after loading a trained adapter.
"""

from .image_dependencies import CORE_PACKAGES, ignore_config_source
from .rollout_image import runtime_image_for

SGLANG_BRANCH = "spindle-sglang-v0.5.21"

# Fork commits in ancestry order, each a descendant of the one before. Append new commits; never edit an entry.
image_01_a3e0e17 = runtime_image_for(
    image="lmsysorg/sglang:v0.5.21",
    branch=SGLANG_BRANCH,
    revision="a3e0e1725406b9330bd062520d4f99d0aa73ee4f",
    core_packages=CORE_PACKAGES,
).add_local_python_source("spindle", ignore=ignore_config_source)
