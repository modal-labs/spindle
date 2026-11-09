"""GLM Flash sampler dependencies; training uses the shared Miles image."""

from .image_dependencies import (
    CORE_PACKAGES,
    ignore_config_source,
)
from .rollout_image import runtime_image_for

# SGLang PR #90 contains GLM dimensions, expert-local reads and async loading.
# Pin its reviewed source together with the matching runtime image. Switch the
# branch to spindle-sglang-v0.5.23 once #90 lands, preserving this exact revision.
SGLANG_IMAGE = "lmsysorg/sglang:v0.5.23"
SGLANG_BRANCH = "codex/spindle-sglang-v0.5.23"
SGLANG_REVISION = "da8bd220df350e44229fb0e7d071e461e69d6d6a"


inference_image = (
    runtime_image_for(
        image=SGLANG_IMAGE,
        branch=SGLANG_BRANCH,
        revision=SGLANG_REVISION,
        # v0.5.23 bundles OTel 1.45.2 and Google API Core requires API >=1.44.
        # Keep that family aligned instead of downgrading only SDK/HTTP to 1.43.
        core_packages=tuple(p.replace("==1.43.0", "==1.45.2") for p in CORE_PACKAGES),
    )
    .env(
        {
            "SPINDLE_GLM_ASYNC_LORA_LOADING": "1",
            "SGLANG_LORA_CSGMV_SPLIT_K_MAX_TOKENS": "128",
        }
    )
    .add_local_python_source("spindle", ignore=ignore_config_source)
)
