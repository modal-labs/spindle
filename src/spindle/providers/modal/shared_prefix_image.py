"""Explicit native-LoRA shared-prefix image; ordinary trainer images are unchanged."""

from .miles_v020_image import grouped_gemm_image_at

# Native shared-prefix PR, based on spindle-miles-v0.2.0.
MILES_COMMIT = "6476fb6b68b8071402f1a2e79ed76d1ce14647ef"
# Both the native parity reference and compact execution use the same TE backend.
image = grouped_gemm_image_at(MILES_COMMIT)
