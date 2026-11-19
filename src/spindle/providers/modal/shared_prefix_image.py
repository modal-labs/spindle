"""Explicit native-LoRA shared-prefix image; ordinary trainer images are unchanged."""

from .miles_v020_image import grouped_gemm_image_at

# Native shared-prefix PR, based on spindle-miles-v0.2.0.
MILES_COMMIT = "e405f377fe3c3de2ea40bc5a86f531e870fd833e"
# Both the native parity reference and compact execution use the same TE backend.
image = grouped_gemm_image_at(MILES_COMMIT)
