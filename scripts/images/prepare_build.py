"""Prepare pinned Miles or SGLang source builds without a Miles registry base."""

import argparse
import hashlib
import json
import re
from pathlib import Path

SHA = re.compile(r"[0-9a-f]{40}")
IMAGE = re.compile(r"[a-z0-9][a-z0-9._:/-]*@sha256:[0-9a-f]{64}")


def replace_required(source: str, old: str, new: str) -> str:
    if old not in source:
        raise ValueError(f"Unsupported upstream Dockerfile; missing {old!r}")
    return source.replace(old, new)


def prepare(
    context: Path,
    *,
    engine: str,
    commit: str,
    base: str,
    uv_base: str = "",
    sglang_commit: str = "",
    megatron_commit: str = "",
) -> dict:
    if not SHA.fullmatch(commit) or not IMAGE.fullmatch(base):
        raise ValueError(
            "Source requires a full commit SHA and base requires a sha256 digest"
        )
    dockerfile = context / "docker/Dockerfile"
    source = dockerfile.read_text()
    args = {}
    if engine == "miles":
        if not all(SHA.fullmatch(pin) for pin in (sglang_commit, megatron_commit)):
            raise ValueError("Miles requires full SGLang and Megatron commit SHAs")
        source = replace_required(
            source,
            "FROM lmsysorg/sglang:${SGLANG_IMAGE_TAG} AS sglang",
            f"FROM {base} AS sglang",
        )
        source = replace_required(
            source,
            "git clone https://github.com/radixark/miles.git",
            "git clone https://github.com/modal-projects/miles.git",
        )
        for name in ("MILES_COMMIT", "SGLANG_COMMIT", "MEGATRON_COMMIT"):
            if not re.search(rf"^ARG {name}=", source, re.MULTILINE):
                raise ValueError(f"Unsupported upstream Dockerfile; no ARG {name}")
        source = replace_required(
            source,
            "git fetch origin ${SGLANG_BRANCH}",
            "git fetch --depth=1 origin ${SGLANG_COMMIT}",
        )
        for path, name in (
            ("Megatron-LM", "MEGATRON_COMMIT"),
            ("miles", "MILES_COMMIT"),
        ):
            checkout = (
                f"git checkout {'-f ' if path == 'Megatron-LM' else ''}${{{name}}}"
            )
            source = replace_required(
                source,
                checkout,
                f"git fetch --depth=1 origin ${{{name}}} && {checkout}",
            )
        args = {
            "MILES_COMMIT": commit,
            "SGLANG_COMMIT": sglang_commit,
            "MEGATRON_COMMIT": megatron_commit,
        }
    elif engine == "sglang":
        if not IMAGE.fullmatch(uv_base):
            raise ValueError("SGLang requires a digest-pinned uv base")
        source = replace_required(
            source,
            "FROM ghcr.io/astral-sh/uv:${UV_VERSION} AS uv",
            f"FROM {uv_base} AS uv",
        )
        source = replace_required(
            source,
            "FROM nvidia/cuda:${CUDA_VERSION}-cudnn-devel-ubuntu24.04 AS",
            f"FROM {base} AS",
        )
        if "ARG BRANCH_TYPE" not in source:
            raise ValueError(
                "Unsupported SGLang Dockerfile; local source mode is unavailable"
            )
        args = {"BRANCH_TYPE": "local", "SGLANG_BUILD_COMMIT": commit}
    else:
        raise ValueError(f"Unknown engine: {engine}")
    dockerfile.write_text(source)
    return {
        "engine": engine,
        "commit": commit,
        "base": base,
        "uv_base": uv_base,
        "build_args": args,
        "dockerfile_sha256": hashlib.sha256(source.encode()).hexdigest(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", choices=("miles", "sglang"), required=True)
    parser.add_argument("--context", type=Path, required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--base", required=True)
    parser.add_argument("--uv-base", default="")
    parser.add_argument("--sglang-commit", default="")
    parser.add_argument("--megatron-commit", default="")
    parser.add_argument("--github-output", type=Path, required=True)
    args = vars(parser.parse_args())
    output = args.pop("github_output")
    metadata = prepare(**args)
    Path("build-inputs.json").write_text(json.dumps(metadata, indent=2) + "\n")
    with output.open("a") as file:
        file.write("build_args<<BUILD_ARGS\n")
        for name, value in metadata["build_args"].items():
            file.write(f"{name}={value}\n")
        file.write("BUILD_ARGS\n")


if __name__ == "__main__":
    main()
