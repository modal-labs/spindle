"""Explicit native-LoRA shared-prefix image; ordinary trainer images are unchanged."""

from .miles_v020_image import grouped_gemm_image_at

# Native shared-prefix PR, based on spindle-miles-v0.2.0.
MILES_COMMIT = "26712cbf8ad733eb6b44740617dc1a4b7102c1ed"
# Both the native parity reference and compact execution use the same TE backend.
image = grouped_gemm_image_at(MILES_COMMIT)
