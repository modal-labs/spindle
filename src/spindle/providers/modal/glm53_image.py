"""GLM Flash sampler settings; training uses the GLM-specific Miles image."""

from .image_dependencies import ignore_config_source
from .rollout_image import runtime_image

inference_image = runtime_image.env(
    {
        "SPINDLE_GLM_ASYNC_LORA_LOADING": "1",
        "SGLANG_LORA_CSGMV_SPLIT_K_MAX_TOKENS": "128",
    }
).add_local_python_source("spindle", ignore=ignore_config_source)
