"""GLM Flash sampler settings; training uses the GLM-specific Miles image."""

from .image_dependencies import ignore_config_source
from .rollout_image import runtime_image_01_c4182b2, runtime_image_05_884f3978

# Fork commits in ancestry order, each a descendant of the one before. Append new commits; never edit an entry.
image_01_c4182b2 = runtime_image_01_c4182b2.env(
    {
        "SPINDLE_GLM_ASYNC_LORA_LOADING": "1",
        "SGLANG_LORA_CSGMV_SPLIT_K_MAX_TOKENS": "128",
    }
).add_local_python_source("spindle", ignore=ignore_config_source)


image_02_884f3978 = runtime_image_05_884f3978.env(
    {
        "SPINDLE_GLM_ASYNC_LORA_LOADING": "1",
        "SGLANG_LORA_CSGMV_SPLIT_K_MAX_TOKENS": "128",
    }
).add_local_python_source("spindle", ignore=ignore_config_source)
