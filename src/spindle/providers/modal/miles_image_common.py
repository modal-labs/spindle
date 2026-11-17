import modal

from .image_dependencies import CORE_PACKAGES, STITCH_PACKAGE


MILES_EXTRA_PACKAGES = (
    "pyyaml>=6.0.2",
    "opentelemetry-exporter-otlp==1.43.0",
    "opentelemetry-exporter-otlp-proto-grpc==1.43.0",
    "opentelemetry-exporter-otlp-proto-http==1.43.0",
)


def miles_base_image(base_image: str, env: dict[str, str]) -> modal.Image:
    return (
        modal.Image.from_registry(base_image)
        .entrypoint([])
        .env(env)
        .pip_install(*CORE_PACKAGES, STITCH_PACKAGE)
    )
