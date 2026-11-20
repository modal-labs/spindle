"""Flat reference catalog, resolved deployment selection and local comparisons."""

import importlib.util
import json
import multiprocessing
import os
from multiprocessing.synchronize import Event
import runpy
import sys
import subprocess
import tomllib
from pathlib import Path
from types import SimpleNamespace

import httpx
import numpy as np
import pytest
import tinker
import toml
from pydantic import BaseModel, ValidationError

from spindle.configuration import BaseConfig
from spindle.deployments import load
from spindle.patching import same_signature

ROOT = Path(__file__).parents[1] / "reference_runs"
sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location("reference_launcher", ROOT / "run.py")
assert spec and spec.loader
launcher = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = launcher
spec.loader.exec_module(launcher)
plotter = runpy.run_path(str(ROOT / "plot.py"))


def test_proxy_wrapper_supports_scripts_with_sibling_imports(tmp_path: Path) -> None:
    folder = tmp_path / "scripts"
    folder.mkdir()
    (folder / "helper.py").write_text("VALUE = 23\n")
    script = folder / "workload.py"
    script.write_text("from helper import VALUE\nprint(VALUE)\n")
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT.parent / "examples/shared-server/with_proxy_auth.py"),
            str(script),
        ],
        cwd=tmp_path,
        env={
            **os.environ,
            "MODAL_PROXY_TOKEN_ID": "test",
            "MODAL_PROXY_TOKEN_SECRET": "test",
        },
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.strip() == "23"


def _record_worker(index: str, entry: dict, ready: Event, release: Event) -> None:
    with launcher.record_run(Path(index), launcher.ReferenceRun(**entry)) as output:
        output.write_text('{"event":"completed"}\n')
        ready.set()
        assert release.wait(10)


@pytest.mark.parametrize("collision", ["id", "output", None])
def test_recording_concurrent_runs_preserves_index(
    collision: str | None, tmp_path: Path
) -> None:
    index = tmp_path / "index.toml"
    index.write_text("# keep provenance comments\nruns = []\n")
    entry = {
        "id": "first",
        "validation": "test",
        "output": "first.jsonl",
        "spindle_commit": "a" * 40,
        "trainer_image": "test-image",
        "miles_commit": "b" * 40,
        "tinker_sdk": "0.26.0",
        "hardware": "test hardware",
    }
    context = multiprocessing.get_context("fork")
    ready, release = context.Event(), context.Event()
    worker = context.Process(
        target=_record_worker, args=(str(index), entry, ready, release)
    )
    worker.start()
    try:
        assert ready.wait(10)
        second = {**entry, "id": "second", "output": "second.jsonl"}
        if collision is not None:
            second[collision] = entry[collision]
            with pytest.raises(ValueError, match="Another run holds"):
                with launcher.record_run(index, launcher.ReferenceRun(**second)):
                    pytest.fail("Colliding run entered workload")
        else:
            with launcher.record_run(index, launcher.ReferenceRun(**second)) as output:
                output.write_text('{"event":"completed"}\n')
    finally:
        release.set()
        worker.join(10)
        if worker.is_alive():
            worker.kill()
            worker.join()
    assert worker.exitcode == 0
    result = plotter["ReferenceRunIndex"].model_validate(
        tomllib.loads(index.read_text())
    )
    assert {run.id for run in result.runs} == (
        {"first", "second"} if collision is None else {"first"}
    )
    assert index.read_text().startswith("# keep provenance comments")
    with pytest.raises(ValueError, match="already indexed"):
        with launcher.record_run(index, launcher.ReferenceRun(**entry)):
            pytest.fail("Completed run was overwritten")


def test_failed_run_keeps_partial_metrics_without_indexing(tmp_path: Path) -> None:
    index = tmp_path / "index.toml"
    entry = launcher.ReferenceRun(
        id="failed",
        validation="test",
        output="partial.jsonl",
        spindle_commit="a" * 40,
        trainer_image="test",
        miles_commit="b" * 40,
        tinker_sdk="0.26.0",
        hardware="test",
    )
    with pytest.raises(RuntimeError, match="workload failed"):
        with launcher.record_run(index, entry) as output:
            output.write_text('{"event":"run"}\n')
            raise RuntimeError("workload failed")
    assert not index.exists()
    assert (tmp_path / "partial.jsonl").read_text() == '{"event":"run"}\n'
    with pytest.raises(FileExistsError):
        with launcher.record_run(index, entry):
            pytest.fail("Partial metrics were overwritten")


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


@pytest.mark.parametrize("location", ["catalog", "scratch", "escape"])
@pytest.mark.parametrize("custom_index", [True, False])
def test_prepare_and_launch_share_overrides_from_another_directory(
    location: str, custom_index: bool, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    catalog_root = tmp_path / "catalog"
    folder = (
        catalog_root / "configs" if location == "catalog" else tmp_path / "definition"
    )
    folder.mkdir(parents=True)
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
    assert prepared.miles_cfg["max_lora_slots"] == 2
    assert prepared.miles_cfg["model_type"] == "qwen3.5-4B"
    assert load(launcher.config_path("qwen35-4b-lora-32k")).name == "qwen35-4b-lora-32k"
    with pytest.raises(FileExistsError):
        launcher.main()

    monkeypatch.setattr(launcher, "__file__", str(catalog_root / "run.py"))
    indexed = location == "catalog"
    output = (catalog_root / "outputs" if indexed else tmp_path) / "run.jsonl"
    if location == "escape":
        (catalog_root / "outputs").mkdir(parents=True)
        (catalog_root / "outputs" / "scratch").symlink_to(
            tmp_path, target_is_directory=True
        )
        output = catalog_root / "outputs" / "scratch" / "run.jsonl"
    monkeypatch.setattr(
        launcher,
        "server_metadata",
        lambda: {"spindle_commit": "c" * 40, "spindle_version": "0.1.8"},
    )
    monkeypatch.setattr(tinker, "reference_test", "sdk-patch-preserved", raising=False)
    index = tmp_path / "index.toml" if custom_index else catalog_root / "index.toml"
    if not indexed:
        index.parent.mkdir(parents=True, exist_ok=True)
        index.write_text("scratch runs must not read or modify this catalog")
    arguments = ["run", str(definition), "--output", str(output)]
    if custom_index:
        arguments.extend(["--index", str(index)])
    monkeypatch.setattr(sys, "argv", arguments)
    before_path, before_argv = sys.path[:], sys.argv
    launcher.main()
    args, value, sdk_patch = json.loads(output.read_text())
    assert args == [
        "--base-model",
        prepared.name,
        "--output",
        str(output.resolve()),
        "--tokenizer",
        "test/tokenizer",
        revision,
        "--literal",
        "a value with spaces",
    ]
    assert (value, sdk_patch) == (23, "sdk-patch-preserved")
    assert sys.path == before_path and sys.argv is before_argv

    if not indexed:
        assert index.read_text() == "scratch runs must not read or modify this catalog"
        with pytest.raises(FileExistsError):
            launcher.main()
        return

    entry = (
        plotter["ReferenceRunIndex"]
        .model_validate(tomllib.loads(index.read_text()))
        .runs[0]
    )
    assert (entry.id, entry.validation, entry.output) == (
        "run",
        "run",
        "catalog/outputs/run.jsonl" if custom_index else "outputs/run.jsonl",
    )
    assert entry.spindle_commit == "c" * 40
    assert entry.spindle_version == "0.1.8"
    assert entry.trainer_image == prepared.trainer_image
    assert entry.miles_commit in prepared.trainer_image
    assert "not attested" in entry.notes
    with pytest.raises(ValueError, match="already indexed"):
        launcher.main()


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
    data = tomllib.loads((ROOT / "configs/dapo_qwen35_4b.toml").read_text())
    data["script_args"] = arguments
    definition = tmp_path / "invalid.toml"
    definition.write_text(toml.dumps(data))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run",
            str(definition),
            "--output",
            str(output),
        ],
    )
    with pytest.raises(SystemExit) as error:
        launcher.main()
    assert error.value.code == 2
    assert not output.exists()


@pytest.mark.parametrize("arguments", [["--smoke"], ["--", "--smoke"]])
def test_launcher_rejects_extra_cli_arguments(
    arguments: list[str],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    output = tmp_path / "metrics.jsonl"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run",
            str(ROOT / "configs/dapo_qwen35_4b.toml"),
            "--output",
            str(output),
            *arguments,
        ],
    )
    with pytest.raises(SystemExit) as error:
        launcher.main()
    assert error.value.code == 2
    assert "unrecognized arguments" in capsys.readouterr().err
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
    }
    if tokenizer is not None:
        data["tokenizer"] = tokenizer
    with pytest.raises(ValidationError, match="tokenizer"):
        launcher.ValidationRunConfig.model_validate(data)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("trainer_gup", "H100!"),
        ("trainer_image", "miles-v0.2.0-15afe21"),
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
        assert plotter["read_output"](ROOT / run.output, published=bool(run.source))
    partial = tmp_path / "partial.jsonl"
    records = [
        {
            "event": "update",
            "time": float(step),
            "step": step,
            "responses": 1,
            "reward": 0.5,
            "response_tokens_mean": 4.0,
            "truncated_fraction": 0.0,
        }
        for step in range(1, 4)
    ]
    records[0]["seconds_since_previous_update"] = 500.0
    records[2]["seconds_since_previous_update"] = 510.0
    partial.write_text("".join(json.dumps(row) + "\n" for row in records) + '{"event":')
    picture = tmp_path / "comparison.png"
    monkeypatch.setattr(
        sys, "argv", ["plot", catalog[0].id, str(partial), "--output", str(picture)]
    )
    runpy.run_path(str(ROOT / "plot.py"), run_name="__main__")
    assert picture.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    interval = plotter["pyplot"].gcf().axes[3].lines[-1]
    assert interval.get_xdata().tolist() == [1, 2, 3]
    assert interval.get_ydata()[0] == 500.0
    assert np.isnan(interval.get_ydata()[1])
    assert interval.get_ydata()[2] == 510.0


def test_plot_keeps_each_run_color_when_another_run_lacks_a_panel(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    execution = tmp_path / "execution.jsonl"
    records = [
        {"event": "evaluation", "time": 0.0, "step": 0},
        {
            "event": "update",
            "time": 1.0,
            "step": 1,
            "seconds_since_previous_update": 1.0,
        },
    ]
    for row in records:
        row.update(
            responses=1, reward=0.5, response_tokens_mean=4.0, truncated_fraction=0.0
        )
    execution.write_text("".join(json.dumps(row) + "\n" for row in records))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "plot",
            "beam-blog-digitized",
            str(execution),
            "--output",
            str(tmp_path / "p.png"),
        ],
    )
    runpy.run_path(str(ROOT / "plot.py"), run_name="__main__")
    axes = plotter["pyplot"].gcf().axes
    assert {axis.lines[-1].get_color() for axis in axes} == {"C1"}


@pytest.mark.parametrize(
    ("field", "value"),
    [("reward", None), ("responses", 0), ("truncated_fraction", 2.0), ("time", "x")],
)
def test_outputs_reject_records_outside_the_schema(
    tmp_path: Path, field: str, value: object
) -> None:
    record = {
        "event": "evaluation",
        "time": 1.0,
        "step": 0,
        "responses": 8,
        "reward": 0.5,
        "response_tokens_mean": 10.0,
        "truncated_fraction": 0.0,
        "script_specific": [1, 2],
        field: value,
    }
    output = tmp_path / "run.jsonl"
    output.write_text(json.dumps(record) + "\n")
    with pytest.raises(ValidationError, match=field):
        plotter["read_output"](output)


def test_index_accepts_spindle_run_without_reported_commit() -> None:
    entry = tomllib.loads((ROOT / "index.toml").read_text())["runs"][0]
    del entry["spindle_commit"]
    assert plotter["ReferenceRunIndex"].model_validate({"runs": [entry]})


@pytest.mark.parametrize(
    ("problem", "message"),
    [
        ("missing_output", "output"),
        ("invalid_commit", "spindle_commit"),
        ("missing_provenance", "missing miles_commit"),
        ("external_with_provenance", "external execution records source"),
        ("external_with_version", "external execution records source"),
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
    elif problem == "missing_provenance":
        del data["runs"][0]["miles_commit"]
    elif problem == "external_with_provenance":
        data["runs"][0]["source"] = "https://example.com/run"
    elif problem == "external_with_version":
        external = next(run for run in data["runs"] if "source" in run)
        external["spindle_version"] = "0.1.8"
    elif problem == "unknown_entry_field":
        data["runs"][0]["trainer_iamge"] = "typo"
    elif problem == "unknown_index_field":
        data["run"] = []
    else:
        data["runs"].append(data["runs"][0].copy())
    with pytest.raises(ValidationError, match=message):
        plotter["ReferenceRunIndex"].model_validate(data)


@pytest.mark.parametrize(
    "validation",
    [
        path
        for path in sorted((ROOT / "configs").glob("*.toml"))
        if tomllib.loads(path.read_text())["script"].endswith("/dapo_math.py")
    ],
)
@pytest.mark.parametrize("restore_case", [None, "optimizer", "scores", "zero_rewards"])
def test_reference_smoke_records_selected_tokenizer_and_updates(
    validation: Path,
    restore_case: str | None,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    workload = importlib.import_module("dapo_math")

    def future(value: object) -> SimpleNamespace:
        return SimpleNamespace(result=lambda: value)

    class Training:
        def __init__(self) -> None:
            self.step = 0
            self.weight = 0.0
            self.moment = 0.0
            self.gradient = 0.0
            self.saved: dict[str, tuple[int, float, float]] = {}

        def save_weights_for_sampler(self, name: str) -> SimpleNamespace:
            return future(SimpleNamespace(path=f"test://{name}"))

        def save_state(self, name: str) -> SimpleNamespace:
            path = f"test://{name}"
            self.saved[path] = self.step, self.weight, self.moment
            return future(SimpleNamespace(path=path))

        def load_state_with_optimizer(self, path: str) -> SimpleNamespace:
            self.step, self.weight, moment = self.saved[path]
            if restore_case != "optimizer":
                self.moment = moment
            return future(None)

        def forward(self, datums: list, loss: str) -> SimpleNamespace:
            value = float("nan") if restore_case == "scores" else self.weight
            return future(
                SimpleNamespace(
                    loss_fn_outputs=[
                        {
                            "logprobs": tinker.types.TensorData(
                                data=[value]
                                * len(datum.loss_fn_inputs["target_tokens"].data),
                                dtype="float32",
                                shape=datum.loss_fn_inputs["target_tokens"].shape,
                            )
                        }
                        for datum in datums
                    ]
                )
            )

        def forward_backward(self, datums: list, loss: str) -> SimpleNamespace:
            assert len(datums) in {8, 32}
            assert loss == "importance_sampling"
            assert all(isinstance(datum, tinker.Datum) for datum in datums)
            advantages = {
                value
                for datum in datums
                for value in datum.loss_fn_inputs["advantages"].data
            }
            if len(datums) == 8:
                assert advantages in ({0.0, 1.0}, {0.0})
            elif restore_case == "zero_rewards":
                assert advantages == {0.0}
            else:
                assert advantages == {-0.5, 0.0, 0.5}
            self.gradient = float(any(advantages))
            return future(SimpleNamespace(metrics={"loss:sum": 0.0}))

        def optim_step(self, adam: object) -> SimpleNamespace:
            self.step += 1
            self.moment = 0.9 * self.moment + self.gradient
            self.weight += 0.01 * self.moment
            return future(SimpleNamespace(metrics={"update_successful:mean": 1.0}))

    class Sampler:
        def sample(self, prompt: object, count: int, params: object) -> SimpleNamespace:
            return future(SimpleNamespace(sequences=(sequences * count)[:count]))

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
        decode=lambda tokens: (
            r"\boxed{1}"
            if tokens[-1] == 4 and restore_case != "zero_rewards"
            else r"\boxed{0}"
        ),
    )
    definition = launcher.ValidationRunConfig.model_validate(
        tomllib.loads(validation.read_text())
    )
    recipe = BaseConfig(
        **vars(load(launcher.config_path(definition.config))),
        overrides=definition.config_overrides.model_dump(
            by_alias=True, exclude_none=True
        ),
    )
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
    smoke = tmp_path / "smoke.toml"
    data = tomllib.loads(validation.read_text())
    data["script"] = str((validation.parent / definition.script).resolve())
    data["script_args"] = [*definition.script_args, "--smoke"]
    smoke.write_text(toml.dumps(data))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run",
            str(smoke),
            "--output",
            str(output),
            "--index",
            str(tmp_path / "index.toml"),
        ],
    )
    if restore_case in {"optimizer", "scores"}:
        error = AssertionError if restore_case == "optimizer" else ValueError
        message = "Not equal" if restore_case == "optimizer" else "nonfinite scores"
        with pytest.raises(error, match=message):
            launcher.main()
        assert not any(
            json.loads(line)["event"] == "restore_validation"
            for line in output.read_text().splitlines()
        )
        return
    launcher.main()
    records = plotter["read_output"](output)
    assert records[0]["config"]["model"] == expected_model
    assert records[0]["config"]["tokenizer_revision"] == definition.tokenizer.revision
    assert records[0]["config"]["updates"] == 2
    assert records[0]["recipe"] == definition.config_overrides.name
    assert [r["step"] for r in records if r["event"] == "update"] == [1, 2]
    assert training.step == 2
    assert (training.step, training.weight, training.moment) == training.saved[
        "test://step-2"
    ]
    restored = records[-1]
    assert restored["event"] == "restore_validation"
    assert restored["method"] == "momentum_replay"
    assert restored["measured_restore_max_abs_error"] == 0.0
    assert restored["restore_max_abs_error"] == restored["replay_max_abs_error"] == 0.0
    assert restored["update_max_abs_change"] > 0
    assert restored["sampled_tokens"] > 0


@pytest.mark.parametrize(
    "reported",
    [
        {"spindle_commit": "d" * 40, "spindle_version": "0.1.8"},
        {"spindle_version": "0.1.8"},
        {},
        {"spindle_commit": None, "spindle_version": ""},
        {"spindle_commit": 123, "spindle_version": ["invalid"]},
    ],
)
def test_server_metadata_uses_stock_sdk_raw_response(
    reported: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    @same_signature(httpx.AsyncHTTPTransport.handle_async_request)
    async def handle(
        self: httpx.AsyncHTTPTransport, request: httpx.Request
    ) -> httpx.Response:
        if request.url.path == "/api/v1/client/config":
            return httpx.Response(
                200, json={"use_pyqwest_transport": False, "pjwt_auth_enabled": False}
            )
        if request.url.path == "/api/v1/client/dynamic_config":
            return httpx.Response(200, json={})
        assert request.url.path == "/api/v1/get_server_capabilities"
        assert request.headers["X-API-Key"] == "tml-test-key"
        assert request.headers["Modal-Key"] == "proxy-id"
        assert request.headers["Modal-Secret"] == "proxy-secret"
        return httpx.Response(200, json={"supported_models": [], **reported})

    monkeypatch.setenv("TINKER_API_KEY", "tml-test-key")
    monkeypatch.setenv("TINKER_BASE_URL", "https://reference.invalid")
    monkeypatch.setenv("TINKER_TELEMETRY", "0")
    monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", handle)
    monkeypatch.setenv("MODAL_PROXY_TOKEN_ID", "proxy-id")
    monkeypatch.setenv("MODAL_PROXY_TOKEN_SECRET", "proxy-secret")
    monkeypatch.setattr(tinker.ServiceClient, "__init__", tinker.ServiceClient.__init__)
    wrapper = runpy.run_path(
        str(ROOT.parent / "examples/shared-server/with_proxy_auth.py")
    )
    wrapper["install"]()
    assert launcher.server_metadata() == {
        key: value if isinstance(value := reported.get(key), str) and value else None
        for key in ("spindle_commit", "spindle_version")
    }


@pytest.mark.parametrize("location", ["outside", "symlink", "extension"])
def test_indexed_outputs_require_catalog_definitions(
    location: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    catalog = tmp_path / "catalog"
    configs = catalog / "configs"
    configs.mkdir(parents=True)
    definition = tmp_path / "dapo_qwen35_4b.toml"
    definition.write_bytes((ROOT / "configs/dapo_qwen35_4b.toml").read_bytes())
    if location == "symlink":
        alias = configs / definition.name
        alias.symlink_to(definition)
        definition = alias
    elif location == "extension":
        definition = definition.rename(configs / "dapo_qwen35_4b.txt")
    output = catalog / "outputs/run.jsonl"
    monkeypatch.setattr(launcher, "__file__", str(catalog / "run.py"))
    monkeypatch.setattr(sys, "argv", ["run", str(definition), "--output", str(output)])
    with pytest.raises(SystemExit) as error:
        launcher.main()
    assert error.value.code == 2
    assert "Indexed outputs require a TOML definition" in capsys.readouterr().err
    assert not output.exists()
    assert not (catalog / "index.toml").exists()


@pytest.mark.parametrize("exit_code", [None, 0, 1])
def test_workload_exit_status_controls_registration(
    exit_code: int | None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    configs = tmp_path / "configs"
    configs.mkdir()
    workload = configs / "workload.py"
    workload.write_text(
        "import sys\nfrom pathlib import Path\n"
        "Path(sys.argv[sys.argv.index('--output') + 1]).write_text('complete metrics\\n')\n"
        f"raise SystemExit({exit_code!r})\n"
    )
    data = tomllib.loads((ROOT / "configs/dapo_qwen35_4b.toml").read_text())
    data["script"] = "workload.py"
    definition = configs / "run.toml"
    definition.write_text(toml.dumps(data))
    output = tmp_path / "outputs/run.jsonl"
    monkeypatch.setattr(launcher, "__file__", str(tmp_path / "run.py"))
    monkeypatch.setattr(
        launcher,
        "server_metadata",
        lambda: {"spindle_commit": None, "spindle_version": None},
    )
    monkeypatch.setattr(sys, "argv", ["run", str(definition), "--output", str(output)])
    if exit_code == 1:
        with pytest.raises(SystemExit) as error:
            launcher.main()
        assert error.value.code == 1
        assert not (tmp_path / "index.toml").exists()
    else:
        launcher.main()
        entry = tomllib.loads((tmp_path / "index.toml").read_text())["runs"][0]
        assert (entry["id"], entry["validation"]) == ("run", "run")
        assert "spindle_commit" not in entry and "spindle_version" not in entry
    assert output.read_text() == "complete metrics\n"


def test_indexed_beam_execution_plots_without_remote_calls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "beam.jsonl"
    output.write_bytes(
        (ROOT / "outputs" / "beam-blog-repro-miles-v012.jsonl").read_bytes()
    )

    def unexpected(**kwargs: object) -> None:
        pytest.fail("plotting created a Tinker client")

    monkeypatch.setattr(tinker, "ServiceClient", unexpected)
    script = ROOT / "scripts" / "beam_blog_repro.py"
    monkeypatch.setattr(
        sys, "argv", [str(script), "--output", str(output), "--plot-only"]
    )
    runpy.run_path(str(script), run_name="__main__")
    summary = json.loads((tmp_path / "beam" / "summary.json").read_text())
    assert summary["accepted_updates"] == 42
    assert summary["evaluations"][-1]["step"] == 40
    assert (tmp_path / "beam" / "learning.png").read_bytes().startswith(b"\x89PNG")
