from copy import deepcopy
import shlex
import subprocess
from types import SimpleNamespace

from spindle.configs.qwen35_9b_lora_16k import Config
from spindle.deployments import DeploymentConfig
from spindle.providers.modal import deployment_apps
from spindle.providers.modal import rollout_image


def test_custom_sampler_image_changes_deployment_identity():
    before = DeploymentConfig.create(Config())
    after = deepcopy(before)
    after.recipe.inference_image = "custom.images:inference"
    assert after.inference_identity() != before.inference_identity()
    assert after.trainer_identity() == before.trainer_identity()


def test_custom_image_is_resolved_on_deployer_only(monkeypatch):
    selected = object()
    imported = []

    def import_module(name):
        imported.append(name)
        return SimpleNamespace(inference=selected)

    monkeypatch.setattr(deployment_apps.importlib, "import_module", import_module)
    monkeypatch.setattr(deployment_apps.modal, "is_local", lambda: True)
    assert deployment_apps.image_for("sglang", "custom.images:inference") is selected
    monkeypatch.setattr(deployment_apps.modal, "is_local", lambda: False)
    deployment_apps.image_for("sglang", "custom.images:inference")
    assert imported == ["custom.images"]


def test_custom_sampler_images_are_built_once_each(monkeypatch):
    built = []
    app = object()
    monkeypatch.setattr(deployment_apps.modal.App, "lookup", lambda *a, **k: app)
    monkeypatch.setattr(
        deployment_apps,
        "image_for",
        lambda backend, name: SimpleNamespace(
            build=lambda target: built.append((name, target))
        ),
    )
    deployment_apps.build_sampler_image(
        "example", "dev", ("", "custom:inference", "custom:inference")
    )
    assert built == [("", app), ("custom:inference", app)]


def test_sampler_checks_out_pinned_revision_on_selected_branch(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.mkdir()

    def git(*args):
        return subprocess.check_output(
            ["git", "-C", str(source), *args], text=True, stderr=subprocess.DEVNULL
        ).strip()

    git("init", "-b", "main")
    git("config", "user.name", "Test")
    git("config", "user.email", "test@example.invalid")
    (source / "model").write_text("base")
    git("add", ".")
    git("commit", "-m", "base")
    git("checkout", "-b", "feature")
    (source / "model").write_text("pinned")
    git("commit", "-am", "feature")
    revision = git("rev-parse", "HEAD")
    (source / "model").write_text("newer")
    git("commit", "-am", "advance feature")
    git("checkout", "main")

    commands = []

    class Build:
        def __getattr__(self, name):
            def record(*args, **kwargs):
                if name == "run_commands":
                    commands.extend(args)
                return self

            return record

    monkeypatch.setattr(rollout_image.modal.Image, "from_registry", lambda *a: Build())
    monkeypatch.setattr(rollout_image, "SGLANG_REPOSITORY", source.as_uri())
    rollout_image.runtime_image_for(
        image="unused", branch="feature", revision=revision, core_packages=()
    )
    checkout = tmp_path / "checkout"
    subprocess.run(
        commands[0].replace("/tmp/spindle-sglang-overlay", shlex.quote(str(checkout))),
        shell=True,
        check=True,
        capture_output=True,
    )
    assert (checkout / "model").read_text() == "pinned"
    assert (
        subprocess.check_output(
            ["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True
        ).strip()
        == revision
    )
