"""Explicit native-LoRA shared-prefix image; ordinary trainer images are unchanged."""

from .miles_v020_image import image_at

# Native shared-prefix PR, based on spindle-miles-v0.2.0.
MILES_COMMIT = "4cc42d18366d4ae9e88b23e3c607ef412165cd49"
image = image_at(MILES_COMMIT)
