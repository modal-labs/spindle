"""Kimi K3 sampler image: SGLang with Kimi K3 LoRA serving on Marlin MXFP4 experts."""

from .image_dependencies import ignore_config_source
from .rollout_image import CORE_SAMPLER_PACKAGES, runtime_image_for

SGLANG_IMAGE = "lmsysorg/sglang:v0.5.23"
SGLANG_BRANCH = "spindle-sglang-v0.5.23"
SGLANG_REVISION = "bb327efd239939c43e2be2c4732428c67d86ffb9"

inference_image = runtime_image_for(
    image=SGLANG_IMAGE,
    branch=SGLANG_BRANCH,
    revision=SGLANG_REVISION,
    core_packages=CORE_SAMPLER_PACKAGES,
).add_local_python_source("spindle", ignore=ignore_config_source)
