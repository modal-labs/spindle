"""Supply Modal secrets to a local Tinker script without printing credentials.

MODAL_PROFILE=prod uv run --no-sync modal run --env dev reference_runs/modal_client.py \
  --base-url "$SPINDLE_URL" --command 'reference_runs/run.py ...'
"""

import os
import shlex
import subprocess
import sys

import modal

app = modal.App("reference-run-client")


@app.function(
    image=modal.Image.debian_slim(python_version="3.12"),
    secrets=[
        modal.Secret.from_name("spindle-api"),
        modal.Secret.from_name("spindle-proxy"),
    ],
)
def credentials() -> dict[str, str]:
    return {
        name: os.environ[name]
        for name in (
            "TINKER_API_KEY",
            "MODAL_PROXY_TOKEN_ID",
            "MODAL_PROXY_TOKEN_SECRET",
        )
    }


@app.local_entrypoint()
def main(base_url: str, command: str) -> None:
    subprocess.run(
        [
            sys.executable,
            "examples/shared-server/with_proxy_auth.py",
            *shlex.split(command),
        ],
        env={**os.environ, **credentials.remote(), "TINKER_BASE_URL": base_url},
        check=True,
    )
