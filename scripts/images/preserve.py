"""Copy complete OCI images and emit deployment pins only after verification."""

import argparse
import hashlib
import pprint
import re
import runpy
import subprocess
import tempfile
from pathlib import Path

LOCK = (
    Path(__file__).resolve().parents[2]
    / "src/spindle/providers/modal/runtime_images.py"
)


def repository(reference: str) -> str:
    return re.sub(r":[^/:]+$", "", reference.split("@")[0])


def manifest_digest(reference: str, *, anonymous: bool = False) -> str:
    command = ["skopeo", "inspect", "--raw", "--retry-times", "3"]
    if anonymous:
        command.append("--no-creds")
    if "@" in reference:
        reference = repository(reference) + "@" + reference.rsplit("@", 1)[1]
    raw = subprocess.check_output([*command, f"docker://{reference}"])
    digest = "sha256:" + hashlib.sha256(raw).hexdigest()
    if "@" in reference and reference.rsplit("@", 1)[1] != digest:
        raise ValueError(f"Registry returned a different manifest for {reference}")
    return digest


def preserve(images: dict, namespace: str, *, repository_prefix: str = "") -> dict:
    if not re.fullmatch(r"[a-z0-9.-]+(?::[0-9]+)?/[a-z0-9._/-]+", namespace):
        raise ValueError("Use a lowercase registry/namespace without a tag or digest")
    if repository_prefix and not re.fullmatch(
        r"[a-z0-9][a-z0-9_.-]*", repository_prefix
    ):
        raise ValueError("Invalid repository prefix")
    result = {}
    for name, entry in images.items():
        if not re.fullmatch(r"[a-z0-9][a-z0-9_.-]*", name):
            raise ValueError(f"Invalid image name: {name}")
        source = entry["image"]
        digest = manifest_digest(source)
        pinned_source = repository(source) + "@" + digest
        target_repository = f"{namespace}/{repository_prefix}{name}"
        tag = f"{target_repository}:preserved-{digest.removeprefix('sha256:')}"
        subprocess.run(
            [
                "skopeo",
                "copy",
                "--all",
                "--preserve-digests",
                "--retry-times",
                "3",
                f"docker://{pinned_source}",
                f"docker://{tag}",
            ],
            check=True,
        )
        target = f"{target_repository}@{digest}"
        result[name] = {**entry, "image": target}
        print(f"Copied {name}: {target}", flush=True)
    for name, entry in result.items():
        manifest_digest(entry["image"], anonymous=True)
        print(f"Verified anonymous pull for {name}", flush=True)
    return result


def write_lock(images: dict, output: Path) -> None:
    source = (
        '"""Base image provenance and deployment pins; see docs/runtime-images.md."""\n\n'
        "IMAGES = "
        + pprint.pformat(images, sort_dicts=False, width=100)
        + '\n\nBASE_IMAGES = {name: entry["image"] for name, entry in IMAGES.items()}\n'
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=output.parent, delete=False) as file:
        file.write(source)
    Path(file.name).replace(output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock", type=Path, default=LOCK)
    parser.add_argument("--namespace", required=True)
    parser.add_argument("--repository-prefix", default="")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    images = runpy.run_path(str(args.lock))["IMAGES"]
    write_lock(
        preserve(images, args.namespace, repository_prefix=args.repository_prefix),
        args.output,
    )


if __name__ == "__main__":
    main()
