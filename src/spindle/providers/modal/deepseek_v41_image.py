"""DeepSeek-V4.1-Flash samplers stay on SGLang v0.5.21.

On v0.5.23, a Flash LoRA sampler hits a GPU memory fault on its first batch
after loading a trained adapter.
"""

from .image_dependencies import CORE_PACKAGES, ignore_config_source
from .rollout_image import runtime_image_for
from .runtime_images import BASE_IMAGES

inference_image = runtime_image_for(
    image=BASE_IMAGES["sglang-v0.5.21"],
    branch="spindle-sglang-v0.5.21",
    revision="a3e0e1725406b9330bd062520d4f99d0aa73ee4f",
    core_packages=CORE_PACKAGES,
).add_local_python_source("spindle", ignore=ignore_config_source)
