"""Flat reference catalog, resolved deployment selection and local comparisons."""

import importlib.util
import json
import runpy
import sys
import tomllib
from pathlib import Path

import pytest
import tinker

from spindle.deployments import load

ROOT = Path(__file__).parents[1] / "reference_runs"
spec = importlib.util.spec_from_file_location("reference_launcher", ROOT / "run.py")
assert spec and spec.loader
launcher = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = launcher
spec.loader.exec_module(launcher)


def test_prepare_and_launch_share_overrides_from_another_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    folder = tmp_path / "definition"
    folder.mkdir()
    (folder / "helper.py").write_text("VALUE = 23\n")
    (folder / "workload.py").write_text(
        "import json, sys, tinker\n"
        "from pathlib import Path\n"
        "from helper import VALUE\n"
        "output = sys.argv[sys.argv.index('--output') + 1]\n"
        "Path(output).write_text(json.dumps([sys.argv[1:], VALUE, tinker.reference_test]))\n"
    )
    definition = folder / "run.toml"
    definition.write_text(
        'config = "qwen35-4b-lora-32k"\nscript = "workload.py"\n'
        'script_args = ["--literal", "a value with spaces"]\n'
        '[config_overrides]\nname = "reference-test"\ntrainer_gpu = "H100!"\n'
        '"miles_cfg.max_lora_slots" = 2\n'
    )
    deployment = tmp_path / "deployment.py"
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        sys, "argv", ["run", str(definition), "--write-config", str(deployment)]
    )
    launcher.main()
    prepared = load(deployment)
    assert prepared.name == "reference-test"
    assert prepared.trainer_gpu == "H100!"
    assert prepared.miles_cfg["max_lora_slots"] == 2
    assert prepared.miles_cfg["model_type"] == "qwen3.5-4B"
    assert load(launcher.config_path("qwen35-4b-lora-32k")).name == "qwen35-4b-lora-32k"
    with pytest.raises(FileExistsError):
        launcher.main()

    output = tmp_path / "run.jsonl"
    monkeypatch.setattr(tinker, "reference_test", "sdk-patch-preserved", raising=False)
    monkeypatch.setattr(
        sys, "argv", ["run", str(definition), "--output", str(output), "--", "--smoke"]
    )
    before_path, before_argv = sys.path[:], sys.argv
    launcher.main()
    args, value, sdk_patch = json.loads(output.read_text())
    assert args == [
        "--base-model",
        prepared.name,
        "--output",
        str(output),
        "--literal",
        "a value with spaces",
        "--smoke",
    ]
    assert (value, sdk_patch) == (23, "sdk-patch-preserved")
    assert sys.path == before_path and sys.argv is before_argv


@pytest.mark.parametrize(
    "arguments", [["--base-model", "wrong"], ["--output=other.jsonl"]]
)
def test_script_arguments_cannot_replace_selected_target_or_output(
    arguments: list[str], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "run.jsonl"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run",
            str(ROOT / "configs/dapo_qwen35_4b.toml"),
            "--output",
            str(output),
            "--",
            *arguments,
        ],
    )
    with pytest.raises(SystemExit) as error:
        launcher.main()
    assert error.value.code == 2
    assert not output.exists()


def test_catalog_outputs_plot_as_ids_and_partial_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    catalog = tomllib.loads((ROOT / "index.toml").read_text())["runs"]
    assert len({run["id"] for run in catalog}) == len(catalog)
    for run in catalog:
        definition = ROOT / "configs" / f"{run['validation']}.toml"
        launcher.ValidationRunConfig.model_validate(
            tomllib.loads(definition.read_text())
        )
        assert (ROOT / run["output"]).is_file()
    partial = tmp_path / "partial.jsonl"
    partial.write_text('{"event":"update","step":1,"reward":0.5}\n{"event":')
    picture = tmp_path / "comparison.png"
    monkeypatch.setattr(
        sys, "argv", ["plot", catalog[0]["id"], str(partial), "--output", str(picture)]
    )
    runpy.run_path(str(ROOT / "plot.py"), run_name="__main__")
    assert picture.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
