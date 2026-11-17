"""Flat reference catalog, resolved deployment selection and local comparisons."""

import importlib.util
import json
import runpy
import sys
import tomllib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import tinker
from pydantic import BaseModel, ValidationError

from spindle.configuration import BaseConfig
from spindle.deployments import DeploymentConfig, load

ROOT = Path(__file__).parents[1] / "reference_runs"
spec = importlib.util.spec_from_file_location("reference_launcher", ROOT / "run.py")
assert spec and spec.loader
launcher = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = launcher
spec.loader.exec_module(launcher)
plotter = runpy.run_path(str(ROOT / "plot.py"))


@pytest.mark.parametrize(
    ("path", "schema"),
    [
        (path, launcher.ValidationRunConfig)
        for path in sorted((ROOT / "configs").glob("*.toml"))
    ]
    + [(path, plotter["ReferenceRunIndex"]) for path in sorted(ROOT.glob("*.toml"))],
    ids=lambda value: (
        str(value.relative_to(ROOT)) if isinstance(value, Path) else value.__name__
    ),
)
def test_committed_toml_matches_schema(path: Path, schema: type[BaseModel]) -> None:
    schema.model_validate(tomllib.loads(path.read_text()))


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
    revision = "a" * 40
    definition.write_text(
        'config = "qwen35-4b-lora-32k"\nscript = "workload.py"\n'
        'trainer_image = "miles-v0.2.0-15afe21"\n'
        'script_args = ["--literal", "a value with spaces"]\n'
        f'tokenizer = {{ model = "test/tokenizer", revision = "{revision}" }}\n'
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
    assert prepared.trainer_image == "miles-v0.2.0-15afe21"
    assert (
        load(launcher.config_path("qwen35-4b-lora-32k")).trainer_image
        == "miles-v0.1.1-fdd808c"
    )
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
        "--tokenizer",
        "test/tokenizer",
        revision,
        "--literal",
        "a value with spaces",
        "--smoke",
    ]
    assert (value, sdk_patch) == (23, "sdk-patch-preserved")
    assert sys.path == before_path and sys.argv is before_argv


@pytest.mark.parametrize(
    "arguments",
    [
        ["--base-model", "wrong"],
        ["--output=other.jsonl"],
        ["--tokenizer", "wrong", "a" * 40],
        ["--tokenizer=wrong"],
    ],
)
def test_script_arguments_cannot_replace_selected_config_values(
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


@pytest.mark.parametrize(
    "tokenizer",
    [
        None,
        {"model": "model"},
        {"model": "model", "revision": "main"},
        {"model": "model", "revision": "a" * 40, "typo": True},
    ],
)
def test_reference_requires_pinned_tokenizer(tokenizer: dict | None) -> None:
    data = {
        "config": "recipe",
        "script": "workload.py",
        "trainer_image": "miles-v0.1.1-fdd808c",
    }
    if tokenizer is not None:
        data["tokenizer"] = tokenizer
    with pytest.raises(ValidationError, match="tokenizer"):
        launcher.ValidationRunConfig.model_validate(data)


@pytest.mark.parametrize("problem", ["missing", "empty", "wrong_type", "override"])
def test_reference_requires_explicit_trainer_image(problem: str) -> None:
    data = tomllib.loads((ROOT / "configs/dapo_qwen35_4b.toml").read_text())
    if problem == "missing":
        del data["trainer_image"]
    elif problem == "empty":
        data["trainer_image"] = ""
    elif problem == "wrong_type":
        data["trainer_image"] = 123
    else:
        data["config_overrides"]["trainer_image"] = "miles-v0.2.0-15afe21"
    with pytest.raises(ValidationError, match="trainer_image"):
        launcher.ValidationRunConfig.model_validate(data)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("trainer_gup", "H100!"),
        ("miles_cfg.cli_options", {}),
        ("trainer_nodes", "2"),
        ("miles_cfg.max_lora_slots", "1"),
    ],
)
def test_reference_rejects_undeclared_or_mistyped_overrides(
    field: str, value: object
) -> None:
    data = tomllib.loads((ROOT / "configs/dapo_qwen35_4b.toml").read_text())
    data["config_overrides"][field] = value
    with pytest.raises(ValidationError, match="config_overrides"):
        launcher.ValidationRunConfig.model_validate(data)


def test_direct_workload_requires_tokenizer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    output = tmp_path / "metrics.jsonl"
    monkeypatch.setattr(
        sys, "argv", ["dapo_math", "--base-model", "recipe", "--output", str(output)]
    )
    with pytest.raises(SystemExit) as error:
        runpy.run_path(str(ROOT / "scripts/dapo_math.py"), run_name="__main__")
    assert error.value.code == 2
    assert "--tokenizer" in capsys.readouterr().err
    assert not output.exists()


def test_catalog_outputs_plot_as_ids_and_partial_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    catalog = (
        plotter["ReferenceRunIndex"]
        .model_validate(tomllib.loads((ROOT / "index.toml").read_text()))
        .runs
    )
    for run in catalog:
        definition = ROOT / "configs" / f"{run.validation}.toml"
        assert definition.is_file()
        assert (ROOT / run.output).is_file()
    partial = tmp_path / "partial.jsonl"
    partial.write_text('{"event":"update","step":1,"reward":0.5}\n{"event":')
    picture = tmp_path / "comparison.png"
    monkeypatch.setattr(
        sys, "argv", ["plot", catalog[0].id, str(partial), "--output", str(picture)]
    )
    runpy.run_path(str(ROOT / "plot.py"), run_name="__main__")
    assert picture.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")


@pytest.mark.parametrize(
    ("problem", "message"),
    [
        ("missing_output", "output"),
        ("invalid_commit", "spindle_commit"),
        ("unknown_entry_field", "extra_forbidden"),
        ("unknown_index_field", "extra_forbidden"),
        ("duplicate_id", "duplicate run IDs in index"),
    ],
)
def test_index_rejects_invalid_provenance(problem: str, message: str) -> None:
    data = tomllib.loads((ROOT / "index.toml").read_text())
    if problem == "missing_output":
        del data["runs"][0]["output"]
    elif problem == "invalid_commit":
        data["runs"][0]["spindle_commit"] = 123
    elif problem == "unknown_entry_field":
        data["runs"][0]["trainer_iamge"] = "typo"
    elif problem == "unknown_index_field":
        data["run"] = []
    else:
        data["runs"].append(data["runs"][0].copy())
    with pytest.raises(ValidationError, match=message):
        plotter["ReferenceRunIndex"].model_validate(data)


@pytest.mark.parametrize("validation", sorted((ROOT / "configs").glob("*.toml")))
def test_reference_smoke_records_selected_tokenizer_and_updates(
    validation: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    workload = importlib.import_module("dapo_math")

    def future(value: object) -> SimpleNamespace:
        return SimpleNamespace(result=lambda: value)

    class Training:
        step = 0

        def save_weights_for_sampler(self, name: str) -> SimpleNamespace:
            return future(SimpleNamespace(path=f"test://{name}"))

        def save_state(self, name: str) -> SimpleNamespace:
            return future(None)

        def forward_backward(self, datums: list, loss: str) -> SimpleNamespace:
            assert len(datums) == 32
            assert loss == "importance_sampling"
            assert all(isinstance(datum, tinker.Datum) for datum in datums)
            assert {
                value
                for datum in datums
                for value in datum.loss_fn_inputs["advantages"].data
            } == {-0.5, 0.0, 0.5}
            return future(SimpleNamespace(metrics={"loss:sum": 0.0}))

        def optim_step(self, adam: object) -> SimpleNamespace:
            self.step += 1
            return future(SimpleNamespace(metrics={"update_successful:mean": 1.0}))

    class Sampler:
        def sample(self, prompt: object, count: int, params: object) -> SimpleNamespace:
            return future(SimpleNamespace(sequences=sequences * (count // 2)))

    training = Training()
    sequences = [
        tinker.types.SampledSequence(
            tokens_np=np.array([3, token], dtype=np.int32),
            logprobs_np=np.array([-1.0, -1.0], dtype=np.float32),
            stop_reason="stop",
            sequence_id=str(token),
        )
        for token in [4, 5]
    ]
    tokenizer = SimpleNamespace(
        eos_token_id=0,
        apply_chat_template=lambda *a, **kw: [1, 2],
        decode=lambda tokens: r"\boxed{1}" if tokens[-1] == 4 else r"\boxed{0}",
    )
    definition = launcher.ValidationRunConfig.model_validate(
        tomllib.loads(validation.read_text())
    )
    recipe = BaseConfig(
        **vars(load(launcher.config_path(definition.config))),
        overrides={
            **definition.config_overrides.model_dump(by_alias=True, exclude_none=True),
            "trainer_image": definition.trainer_image,
        },
    )
    deployment = DeploymentConfig.create(recipe)
    assert (
        deployment.trainer_settings["miles"]["actor_num_nodes"] == recipe.trainer_nodes
    )
    if recipe.trainer_nodes > 1:
        assert deployment.trainer_settings["capture_dir"].startswith("/checkpoints/")
    expected_model = recipe.model

    def load_tokenizer(model: str, *, revision: str) -> SimpleNamespace:
        assert model == expected_model == definition.tokenizer.model
        assert revision == definition.tokenizer.revision
        return tokenizer

    monkeypatch.setattr(
        workload.PreTrainedTokenizerFast, "from_pretrained", load_tokenizer
    )
    dataset = tmp_path / "dataset.jsonl"
    dataset.write_text((json.dumps({"prompt": [], "label": "1"}) + "\n") * 32)
    monkeypatch.setattr(
        "huggingface_hub.hf_hub_download", lambda *a, **kw: str(dataset)
    )
    service = SimpleNamespace(
        create_lora_training_client=lambda model, rank: training,
        create_sampling_client=lambda model_path: Sampler(),
    )
    monkeypatch.setattr(tinker, "ServiceClient", lambda: service)
    output = tmp_path / "metrics.jsonl"
    monkeypatch.setattr(
        sys, "argv", ["run", str(validation), "--output", str(output), "--", "--smoke"]
    )
    launcher.main()
    records = [json.loads(line) for line in output.read_text().splitlines()]
    assert records[0]["config"]["model"] == expected_model
    assert records[0]["config"]["tokenizer_revision"] == definition.tokenizer.revision
    assert records[0]["config"]["updates"] == 2
    assert records[0]["recipe"] == definition.config_overrides.name
    assert [r["step"] for r in records if r["event"] == "update"] == [1, 2]
    assert training.step == 2
