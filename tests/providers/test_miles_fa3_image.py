import hashlib
import json

import pytest

from spindle.providers.modal.miles_fa3_image import (
    FA3_REVISION,
    WHEEL,
    validate_artifacts,
)


@pytest.fixture
def artifacts(tmp_path):
    payload = b"test wheel contents"
    (tmp_path / WHEEL).write_bytes(payload)
    manifest = {
        "revision": FA3_REVISION,
        "name": WHEEL,
        "binary_sha256": {"flash_attn_3.so": "example"},
        "wheel_sha256": hashlib.sha256(payload).hexdigest(),
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    return tmp_path


def test_accepts_matching_artifact_manifest(artifacts):
    validate_artifacts(artifacts)


def test_rejects_replaced_wheel_before_building_image(artifacts):
    (artifacts / WHEEL).write_bytes(b"different build")
    with pytest.raises(ValueError, match="does not match"):
        validate_artifacts(artifacts)


@pytest.mark.parametrize(
    ("key", "value", "error"),
    [
        ("revision", "other", "pinned source revision"),
        ("name", "other.whl", "wheel name"),
        ("binary_sha256", {}, "CUDA binaries"),
    ],
)
def test_rejects_incompatible_manifest(artifacts, key, value, error):
    path = artifacts / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest[key] = value
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match=error):
        validate_artifacts(artifacts)


def test_experimental_image_requires_explicit_artifacts(monkeypatch):
    from spindle.providers.modal.deployment_apps import image_for

    monkeypatch.delenv("SPINDLE_MILES_FA3_ARTIFACTS", raising=False)
    with pytest.raises(ValueError, match="requires SPINDLE_MILES_FA3_ARTIFACTS"):
        image_for("miles", "miles-deterministic-fa3")


def test_artifact_builder_uses_the_current_pinned_base_image():
    import ast
    from pathlib import Path
    from spindle.providers.modal.miles_image import BASE_IMAGE

    source = Path(__file__).parents[2] / "scripts/run_fa3_deterministic.py"
    module = ast.parse(source.read_text())
    assignment = next(
        node
        for node in module.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "BASE_IMAGE"
            for target in node.targets
        )
    )
    assert ast.literal_eval(assignment.value) == BASE_IMAGE
