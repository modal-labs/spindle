"""Opt-in sampler images for adapter-scoped policy migration."""

from .image_dependencies import ignore_config_source
from .rollout_image import CORE_SAMPLER_PACKAGES, SGLANG_IMAGE, runtime_image_for

SGLANG_BRANCH = "codex/adapter-scoped-update"

runtime_image_01_884f397 = runtime_image_for(
    image=SGLANG_IMAGE,
    branch=SGLANG_BRANCH,
    revision="884f397815f8b3b6b2d98fd2a54590d16d04adc9",
    core_packages=CORE_SAMPLER_PACKAGES,
)
image_01_884f397 = runtime_image_01_884f397.add_local_python_source(
    "spindle", ignore=ignore_config_source
)
glm53_image_01_884f397 = runtime_image_01_884f397.env(
    {
        "SPINDLE_GLM_ASYNC_LORA_LOADING": "1",
        "SGLANG_LORA_CSGMV_SPLIT_K_MAX_TOKENS": "128",
    }
).add_local_python_source("spindle", ignore=ignore_config_source)
