import hashlib
import runpy
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

SCRIPTS = Path(__file__).parents[2] / "scripts/images"
RAW_MANIFEST = b'{"schemaVersion":2,"manifests":[{"platform":{"architecture":"amd64"}},{"platform":{"architecture":"arm64"}}]}'
DIGEST = "sha256:" + hashlib.sha256(RAW_MANIFEST).hexdigest()
SOURCE_SHA = "a" * 40
BASE = "ghcr.io/example/sglang@sha256:" + "b" * 64


@pytest.fixture
def preservation():
    return SimpleNamespace(**runpy.run_path(str(SCRIPTS / "preserve.py")))


@pytest.fixture
def preparation():
    return SimpleNamespace(**runpy.run_path(str(SCRIPTS / "prepare_build.py")))


def test_preservation_resolves_tag_once_and_writes_verified_owned_pin(
    preservation, monkeypatch, tmp_path
):
    commands = []

    def inspect(command):
        commands.append(command)
        return RAW_MANIFEST

    def copy(command, *, check):
        assert check
        commands.append(command)

    monkeypatch.setattr(subprocess, "check_output", inspect)
    monkeypatch.setattr(subprocess, "run", copy)
    images = {
        "sglang": {"upstream": "upstream/sglang:v1", "image": "upstream/sglang:v1"}
    }
    preserved = preservation.preserve(
        images, "docker.io/example", repository_prefix="spindle-"
    )
    preservation.write_lock(preserved, tmp_path / "runtime_images.py")
    lock = runpy.run_path(str(tmp_path / "runtime_images.py"))
    assert lock["BASE_IMAGES"]["sglang"] == f"docker.io/example/spindle-sglang@{DIGEST}"
    assert lock["IMAGES"]["sglang"]["upstream"] == "upstream/sglang:v1"
    assert images["sglang"]["image"] == "upstream/sglang:v1"
    copy_command = commands[1]
    assert f"docker://upstream/sglang@{DIGEST}" in copy_command
    assert "--all" in copy_command and "--preserve-digests" in copy_command
    assert "--no-creds" in commands[2]


@pytest.mark.parametrize("failure", ["copy", "private", "changed-manifest"])
def test_failed_preservation_keeps_existing_deployment_lock(
    preservation, monkeypatch, tmp_path, failure
):
    output = tmp_path / "runtime_images.py"
    output.write_text("existing deployment lock")
    inventory = tmp_path / "input.py"
    inventory.write_text("IMAGES = {'test': {'image': 'upstream/test:v1'}}")

    def inspect(command):
        if "--no-creds" in command:
            if failure == "private":
                raise subprocess.CalledProcessError(1, command)
            if failure == "changed-manifest":
                return b'{"schemaVersion":1}'
        return RAW_MANIFEST

    def copy(command, **kwargs):
        if failure == "copy":
            raise subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(subprocess, "check_output", inspect)
    monkeypatch.setattr(subprocess, "run", copy)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "preserve.py",
            "--lock",
            str(inventory),
            "--namespace",
            "ghcr.io/example/runtimes",
            "--output",
            str(output),
        ],
    )
    with pytest.raises((ValueError, subprocess.CalledProcessError)):
        preservation.main()
    assert output.read_text() == "existing deployment lock"


def test_pinned_source_rejects_manifest_mismatch(preservation, monkeypatch):
    monkeypatch.setattr(subprocess, "check_output", lambda command: RAW_MANIFEST)
    with pytest.raises(ValueError, match="different manifest"):
        preservation.manifest_digest("upstream/image@sha256:" + "0" * 64)


def test_preservation_rerun_uses_retained_copy(preservation, monkeypatch):
    sources = []

    def inspect(command):
        sources.append(command[-1])
        return RAW_MANIFEST

    monkeypatch.setattr(subprocess, "check_output", inspect)
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: None)
    preservation.preserve(
        {
            "test": {
                "upstream": "deleted/image:dev",
                "image": f"ghcr.io/example/runtimes/test@{DIGEST}",
            }
        },
        "ghcr.io/example/runtimes",
    )
    assert all("deleted/image" not in reference for reference in sources)


@pytest.mark.parametrize(
    "reference,expected",
    [
        ("docker.io/org/image:tag", "docker.io/org/image"),
        (f"docker.io/org/image:tag@{DIGEST}", "docker.io/org/image"),
        ("localhost:5000/org/image:tag", "localhost:5000/org/image"),
    ],
)
def test_registry_repository_keeps_ports(preservation, reference, expected):
    assert preservation.repository(reference) == expected


def test_miles_build_fetches_exact_revisions_on_a_pinned_sglang_base(
    preparation, tmp_path
):
    dockerfile = tmp_path / "docker/Dockerfile"
    dockerfile.parent.mkdir()
    dockerfile.write_text("""FROM lmsysorg/sglang:${SGLANG_IMAGE_TAG} AS sglang
ARG MILES_COMMIT=main
ARG SGLANG_COMMIT=""
ARG MEGATRON_COMMIT=""
RUN git fetch origin ${SGLANG_BRANCH} && git checkout -f ${SGLANG_COMMIT}
RUN git checkout -f ${MEGATRON_COMMIT}
RUN git clone https://github.com/radixark/miles.git /root/miles && git checkout ${MILES_COMMIT}
""")
    metadata = preparation.prepare(
        tmp_path,
        engine="miles",
        commit=SOURCE_SHA,
        base=BASE,
        sglang_commit="c" * 40,
        megatron_commit="d" * 40,
    )
    source = dockerfile.read_text()
    assert source.startswith(f"FROM {BASE} AS sglang\n")
    assert "git fetch --depth=1 origin ${MILES_COMMIT}" in source
    assert "git fetch --depth=1 origin ${MEGATRON_COMMIT}" in source
    assert "git fetch --depth=1 origin ${SGLANG_COMMIT}" in source
    assert "git clone https://github.com/modal-projects/miles.git" in source
    assert metadata["build_args"]["MILES_COMMIT"] == SOURCE_SHA
    assert metadata["dockerfile_sha256"] == hashlib.sha256(source.encode()).hexdigest()


def test_sglang_build_pins_both_cuda_stages_and_uses_local_source(
    preparation, tmp_path
):
    dockerfile = tmp_path / "docker/Dockerfile"
    dockerfile.parent.mkdir()
    dockerfile.write_text("""FROM ghcr.io/astral-sh/uv:${UV_VERSION} AS uv
FROM nvidia/cuda:${CUDA_VERSION}-cudnn-devel-ubuntu24.04 AS base
ARG BRANCH_TYPE=remote
FROM nvidia/cuda:${CUDA_VERSION}-cudnn-devel-ubuntu24.04 AS runtime
""")
    uv = "ghcr.io/example/uv@sha256:" + "e" * 64
    metadata = preparation.prepare(
        tmp_path, engine="sglang", commit=SOURCE_SHA, base=BASE, uv_base=uv
    )
    source = dockerfile.read_text()
    assert source.count(f"FROM {BASE} AS") == 2
    assert f"FROM {uv} AS uv" in source
    assert metadata["build_args"]["BRANCH_TYPE"] == "local"


@pytest.mark.parametrize("commit,base", [("main", BASE), (SOURCE_SHA, "sglang:latest")])
def test_build_rejects_moving_inputs(preparation, tmp_path, commit, base):
    with pytest.raises(ValueError, match="full commit SHA"):
        preparation.prepare(tmp_path, engine="miles", commit=commit, base=base)


def test_upstream_dockerfile_drift_leaves_source_untouched(preparation, tmp_path):
    dockerfile = tmp_path / "docker/Dockerfile"
    dockerfile.parent.mkdir()
    dockerfile.write_text("FROM unexpected/image:latest\n")
    with pytest.raises(ValueError, match="Unsupported upstream Dockerfile"):
        preparation.prepare(
            tmp_path, engine="sglang", commit=SOURCE_SHA, base=BASE, uv_base=BASE
        )
    assert dockerfile.read_text() == "FROM unexpected/image:latest\n"
