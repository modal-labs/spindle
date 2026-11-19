"""Explicit native-LoRA shared-prefix image; ordinary trainer images are unchanged."""

from .miles_v020_image import grouped_gemm_image_at

# Native shared-prefix PR, based on spindle-miles-v0.2.0.
MILES_COMMIT = "0e2b7fcf39381cba791bc48433febbba6afd79e5"
# Both the native parity reference and compact execution use the same TE backend.
image = grouped_gemm_image_at(MILES_COMMIT)
