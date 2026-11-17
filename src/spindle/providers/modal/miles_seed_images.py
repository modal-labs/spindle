"""Miles images with native per-adapter initialization seeds."""

from .glm53_trainer_image import build_image as glm53_image
from .miles_image import build_image as v011_image
from .miles_v012_image import build_image as v012_image
from .miles_v020_image import build_image as v020_image


IMAGES = {
    "miles-v0.1.1-ce494f0": v011_image("ce494f0259b74fc406629dd44f5467c26e9dcb75"),
    "miles-v0.1.2-612703a": v012_image("612703a07c6a1e569ffa5059a66796cc693d95b1"),
    "miles-v0.1.2-612703a-glm53": glm53_image(
        "612703a07c6a1e569ffa5059a66796cc693d95b1"
    ),
    "miles-v0.2.0-384d73b": v020_image("384d73b13da2295dc9bb094da177bb5f4c41b682"),
}
