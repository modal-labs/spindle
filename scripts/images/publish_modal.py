"""Publish retained bases using a registry secret inside a Modal worker."""

import os
import subprocess
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[2]
app = modal.App("spindle-preserve-runtime-images")
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("skopeo")
    .add_local_file(str(ROOT / "scripts/images/preserve.py"), "/opt/preserve.py")
    .add_local_file(
        str(ROOT / "src/spindle/providers/modal/runtime_images.py"),
        "/opt/runtime_images.py",
    )
)


@app.function(
    image=image,
    secrets=[
        modal.Secret.from_name(
            os.environ.get("SPINDLE_REGISTRY_SECRET", "dockerhub-modalresearch")
        )
    ],
    timeout=21600,
    cpu=2,
    memory=2048,
)
def publish(username: str) -> str:
    username = username or next(
        (
            os.environ[key]
            for key in ("DOCKERHUB_USERNAME", "REGISTRY_USERNAME", "DOCKER_USERNAME")
            if os.environ.get(key)
        ),
        "",
    )
    password = next(
        (
            os.environ[key]
            for key in (
                "DOCKERHUB_TOKEN",
                "DOCKERHUB_PASSWORD",
                "REGISTRY_PASSWORD",
                "DOCKER_PASSWORD",
            )
            if os.environ.get(key)
        ),
        "",
    )
    if not username or not password:
        raise ValueError("Set the registry secret's login fields or pass --username")
    login = subprocess.run(
        ["skopeo", "login", "--username", username, "--password-stdin", "docker.io"],
        input=password,
        text=True,
        capture_output=True,
    )
    if login.returncode:
        raise RuntimeError(
            "Docker Hub login failed; check the configured account/token"
        )
    subprocess.run(
        [
            "python",
            "/opt/preserve.py",
            "--lock",
            "/opt/runtime_images.py",
            "--namespace",
            "docker.io/modalresearch",
            "--repository-prefix",
            "spindle-",
            "--output",
            "/tmp/owned-runtime-images.py",
        ],
        check=True,
    )
    return Path("/tmp/owned-runtime-images.py").read_text()


@app.local_entrypoint()
def main(username: str = "", output: str = "/tmp/owned-runtime-images.py") -> None:
    Path(output).write_text(publish.remote(username))
    print(f"Verified deployment pins saved to {output}")
