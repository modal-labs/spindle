"""Explicit experimental overlay; ordinary Miles trainer images are unchanged."""

import modal

from .glm53_trainer_image import image as glm53_base
from .image_dependencies import ignore_config_source

# Unified shared-prefix implementation, modal-projects/miles#35. Copy only the
# optional package; preserve the base image's Miles/Bridge/Megatron API pins.
PREFIX_CORE_COMMIT = "6c91d704b0568c5707740a1fae477491984b39a1"
PREFIX_CORE_PATH = "miles/backends/training_utils/prefix_tree"


def with_shared_prefix(base: modal.Image) -> modal.Image:
    return (
        base.run_commands(
            "git -C /root/miles fetch --depth=1"
            f" https://github.com/modal-projects/miles.git {PREFIX_CORE_COMMIT}"
            f" && git -C /root/miles checkout FETCH_HEAD -- {PREFIX_CORE_PATH}",
            "python -c 'from miles.backends.training_utils.prefix_tree.packed "
            "import prepare_packed_tree; "
            "from miles.backends.training_utils.prefix_tree.kernel_policy "
            "import LoraKernelControl'",
        )
        .env({"SPINDLE_PREFIX_CORE_COMMIT": PREFIX_CORE_COMMIT})
        .add_local_python_source("spindle", ignore=ignore_config_source)
    )


glm53_trainer_image = with_shared_prefix(glm53_base)
