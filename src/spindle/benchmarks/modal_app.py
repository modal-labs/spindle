"""Proxy-authenticated collector; deploy with --strategy recreate for one writer."""

import os

import modal

from .server import create_app
from spindle.providers.modal.image_dependencies import (
    CORE_PACKAGES,
    STITCH_PACKAGE,
    TINKER_PACKAGE,
)

app = modal.App("spindle-benchmarks")
auth_secret = os.environ.get("SPINDLE_BENCHMARK_AUTH_SECRET")
forward_secret = os.environ.get("SPINDLE_BENCHMARK_FORWARD_SECRET")
collector_secrets = []
if auth_secret:
    collector_secrets.append(
        modal.Secret.from_name(auth_secret, required_keys=["SPINDLE_BENCHMARK_TOKEN"])
    )
if forward_secret:
    collector_secrets.append(
        modal.Secret.from_name(
            forward_secret,
            required_keys=["SPINDLE_BENCHMARK_FORWARD_OTLP_ENDPOINT"],
        )
    )
volume = modal.Volume.from_name("spindle-benchmark-results", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install(
        *CORE_PACKAGES,
        STITCH_PACKAGE,
        TINKER_PACKAGE,
        "huggingface-hub",
    )
)
image = image.add_local_python_source("spindle").add_local_file(
    os.path.join(os.path.dirname(__file__), "dashboard.html"),
    "/root/spindle/benchmarks/dashboard.html",
)


@app.function(
    image=image,
    volumes={"/results": volume},
    secrets=collector_secrets,
    max_containers=1,
)
@modal.concurrent(max_inputs=16, target_inputs=8)
@modal.asgi_app(requires_proxy_auth=True)
def server():
    return create_app(
        "/results",
        token=os.environ.get("SPINDLE_BENCHMARK_TOKEN"),
        persist=volume.commit,
    )
