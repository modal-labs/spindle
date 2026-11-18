"""Explicit native-LoRA shared-prefix image; ordinary trainer images are unchanged."""

from .miles_v020_image import image_at

# Native shared-prefix PR, based on spindle-miles-v0.2.0.
MILES_COMMIT = "42910763ada40056d6e1845f7b0e0a9e938c7a88"
image = image_at(MILES_COMMIT)
