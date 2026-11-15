"""Kimi K3 trainer and sampler images.

The trainer runs Miles v0.2.0 with Kimi K3 LoRA on the MXFP4 release weights.
The sampler runs SGLang with Kimi K3 LoRA serving on Marlin MXFP4 experts.
"""

from .image_dependencies import ignore_config_source
from .miles_v020_image import fork_image
from .rollout_image import CORE_SAMPLER_PACKAGES, runtime_image_for

MILES_BRANCH = "spindle-miles-v0.2.0"
MILES_COMMIT = "74c92c5def32c252edc18fd588ff8aa9776d0792"
SGLANG_IMAGE = "lmsysorg/sglang:v0.5.23"
SGLANG_BRANCH = "spindle-sglang-v0.5.23-kimi-k3"
SGLANG_REVISION = "9acab080bb0717dfa9f74307fd2191e9827d8c5f"

trainer_image = fork_image(MILES_BRANCH, MILES_COMMIT)

inference_image = runtime_image_for(
    image=SGLANG_IMAGE,
    branch=SGLANG_BRANCH,
    revision=SGLANG_REVISION,
    core_packages=CORE_SAMPLER_PACKAGES,
).add_local_python_source("spindle", ignore=ignore_config_source)
