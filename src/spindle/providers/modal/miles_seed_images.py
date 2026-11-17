"""Miles images with native per-adapter initialization seeds."""

import modal

from .glm53_trainer_image import image as glm53_image
from .miles_image import image as v011_image
from .miles_v012_image import image as v012_image
from .miles_v020_image import image as v020_image


def _with_seed_api(image: modal.Image, commit: str) -> modal.Image:
    return image.run_commands(
        f"git -C /root/miles fetch --depth=1 https://github.com/modal-projects/miles.git {commit}"
        " && git -C /root/miles checkout --detach FETCH_HEAD"
    )


IMAGES = {
    "miles-v0.1.1-ce494f0": _with_seed_api(
        v011_image, "ce494f0259b74fc406629dd44f5467c26e9dcb75"
    ),
    "miles-v0.1.2-612703a": _with_seed_api(
        v012_image, "612703a07c6a1e569ffa5059a66796cc693d95b1"
    ),
    "miles-v0.1.2-612703a-glm53": _with_seed_api(
        glm53_image, "612703a07c6a1e569ffa5059a66796cc693d95b1"
    ),
    "miles-v0.2.0-384d73b": _with_seed_api(
        v020_image, "384d73b13da2295dc9bb094da177bb5f4c41b682"
    ),
}
