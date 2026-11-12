import importlib.util
import os
import signal
import subprocess
import base64
import shutil
import time
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest
import httpx
from tinker.lib import internal_client_holder


spec = importlib.util.spec_from_file_location(
    "beam_reference_run", Path(__file__).parents[1] / "reference_runs" / "beam.py"
)
assert spec and spec.loader
beam = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = beam
spec.loader.exec_module(beam)


def test_proxy_headers_reach_the_stock_sdk_transport(tmp_path, monkeypatch):
    provenance = tmp_path / "deployment.json"
    provenance.write_text("{}")
    args = SimpleNamespace(
        output=tmp_path / "results",
        replay_cache=tmp_path / "cache",
        deployment_provenance=provenance,
        resume=None,
        stop_after=None,
    )
    monkeypatch.setattr(
        beam.PreTrainedTokenizerFast, "from_pretrained", lambda *a, **kw: None
    )
    monkeypatch.setattr(
        beam, "load_data", lambda *a: ([({}, [1])] * 3200, [({}, [1])] * 30)
    )
    monkeypatch.setattr(beam, "plot", lambda *a, **kw: None)
    monkeypatch.setenv("TINKER_BASE_URL", "https://example.invalid")
    monkeypatch.setenv("TINKER_API_KEY", "tml-test")
    monkeypatch.setenv("MODAL_KEY", "test-proxy-key")
    monkeypatch.setenv("MODAL_SECRET", "test-proxy-secret")

    def request(request):
        assert request.headers["Modal-Key"] == "test-proxy-key"
        assert request.headers["Modal-Secret"] == "test-proxy-secret"
        return httpx.Response(400, json={"error": "stop_before_model_creation"})

    async_client = internal_client_holder.AsyncTinker
    monkeypatch.setattr(
        internal_client_holder,
        "AsyncTinker",
        lambda **kw: async_client(
            **kw, http_client=httpx.AsyncClient(transport=httpx.MockTransport(request))
        ),
    )
    with pytest.raises(beam.tinker.BadRequestError, match="stop_before_model_creation"):
        beam.run(args, beam.Config())


def test_interrupted_append_recovery_preserves_completed_records(tmp_path):
    path = tmp_path / "metrics.jsonl"
    with beam.Journal(tmp_path) as journal:
        journal.event("accepted", step=1)
        with path.open("ab") as stream:
            stream.write(b'{"event":"interrupted')
        assert [r["step"] for r in beam.read_records(path)] == [1]
    with beam.Journal(tmp_path) as journal:
        journal.event("resumed", step=2)
    records = beam.read_records(path)
    assert [(r["event"], r["step"]) for r in records] == [
        ("accepted", 1),
        ("resumed", 2),
    ]
    assert len(path.read_text().splitlines()) == 2


def test_writer_exclusion_and_release(tmp_path):
    with beam.Journal(tmp_path):
        with pytest.raises(RuntimeError, match="Another reference-run writer"):
            beam.Journal(tmp_path)
    with beam.Journal(tmp_path) as journal:
        journal.event("resumed")
    assert beam.read_records(tmp_path / "metrics.jsonl")[0]["event"] == "resumed"


def test_concurrent_responses_are_all_durable(tmp_path):
    with beam.Journal(tmp_path) as journal:
        with ThreadPoolExecutor(max_workers=8) as pool:
            list(
                pool.map(
                    lambda i: journal.append("eval.jsonl", {"index": i}), range(100)
                )
            )
    records = beam.read_records(tmp_path / "eval.jsonl")
    assert sorted(r["index"] for r in records) == list(range(100))


def test_corrupt_completed_record_is_not_hidden(tmp_path):
    path = tmp_path / "metrics.jsonl"
    path.write_text('{"event":\n')
    with pytest.raises(json.JSONDecodeError):
        beam.read_records(path)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_metrics_leave_existing_results_intact(tmp_path, value):
    with beam.Journal(tmp_path) as journal:
        journal.event("accepted", step=1)
        with pytest.raises(ValueError):
            journal.event("update", loss=value)
    assert len(beam.read_records(tmp_path / "metrics.jsonl")) == 1


@pytest.mark.parametrize(
    "text,label,expected",
    [
        (r"Answer: \boxed{42}", "42", 1),
        (r"\boxed{1} then \boxed{2}", "1", 0),
        (r"\boxed{336}", r"336^\circ", 0),
        (r"\boxed{2}" + "x" * 100, "2", 0),
        (r"\boxed{a{b}}", "a{b}", 1),
        (r"\boxed{42", "42", 0),
    ],
)
def test_strict_reference_grading(text, label, expected):
    assert beam.grade(text, label) == expected


def test_partial_evaluation_survives_failure_and_resumes(tmp_path, monkeypatch):

    config = beam.dataclasses.replace(beam.Config(), samples_per_prompt=2)
    rows = [(dict(prompt=[{"content": "p"}], label="2"), [1])]
    tokenizer = SimpleNamespace(decode=lambda _: "\\boxed{2}")
    sequence = SimpleNamespace(tokens=[2], stop_reason="stop")
    monkeypatch.setattr(
        beam, "sample_group", lambda *a, **kw: ([sequence, sequence], [1.0, 1.0])
    )
    with beam.Journal(tmp_path) as journal:
        journal.append(
            "aime25-step-0000.jsonl",
            dict(
                step=0,
                problem=0,
                sample=0,
                correct=1.0,
                tokens=1,
                stop_reason="stop",
                text="saved response",
            ),
        )
        beam.evaluate(None, rows, 0, config, tokenizer, journal)
    records = beam.read_records(tmp_path / "aime25-step-0000.jsonl")
    assert len(records) == 2
    assert records[0]["text"] == "saved response"
    assert records[1]["sample"] == 1


def test_learning_criterion_uses_final_not_best_checkpoint(tmp_path):
    with beam.Journal(tmp_path) as journal:
        for step, correct in [(0, 0.0), (60, 1.0), (100, 0.0)]:
            for problem in range(30):
                for sample in range(8):
                    journal.append(
                        f"aime25-step-{step:04d}.jsonl",
                        dict(
                            problem=problem, sample=sample, correct=correct, tokens=100
                        ),
                    )
    summary = beam.summarize(tmp_path)
    assert summary["final_comparison"]["learning_demonstrated"] is False
    assert summary["final_comparison"]["accuracy_delta"] == 0


def test_partial_final_evaluation_cannot_claim_learning(tmp_path):
    with beam.Journal(tmp_path) as journal:
        for step in [0, 100]:
            journal.append(
                f"aime25-step-{step:04d}.jsonl",
                dict(problem=0, sample=0, correct=float(step > 0), tokens=100),
            )
    summary = beam.summarize(tmp_path)
    assert summary["final_comparison"] is None
    assert all(not r["complete"] for r in summary["evaluations"])


@pytest.mark.parametrize("stop_after", [None, 9])
def test_invalid_resume_does_not_change_training_history(tmp_path, stop_after):

    output = tmp_path / "results"
    with beam.Journal(output) as journal:
        journal.event("checkpoint", step=10, path="tinker://checkpoint10")
        journal.event("update_started", step=11)
        journal.event("optimizer_submitted", step=11)
    args = SimpleNamespace(
        output=output,
        resume="tinker://checkpoint10",
        replay_cache=tmp_path / "cache",
        stop_after=stop_after,
    )
    message = (
        "Uncheckpointed update outcome"
        if stop_after is None
        else "--stop-after precedes"
    )
    with pytest.raises(ValueError, match=message):
        beam.run(args, beam.Config())
    assert len(beam.read_records(output / "metrics.jsonl")) == 3


@pytest.mark.parametrize("lose_cache", [False, True])
@pytest.mark.parametrize(
    "failure_phase", [None, "prefetch", "publish", "backward", "rejected"]
)
def test_checkpoint_resume_preserves_cached_behavior_and_update_sequence(
    tmp_path, monkeypatch, lose_cache, failure_phase
):

    config = beam.dataclasses.replace(
        beam.Config(), updates=2, groups=2, samples_per_prompt=2
    )
    output, cache = tmp_path / "results", tmp_path / "replay"
    provenance = tmp_path / "deployment.json"
    provenance.write_text("{}")
    args = SimpleNamespace(
        output=output,
        replay_cache=cache,
        deployment_provenance=provenance,
        resume=None,
        stop_after=None if failure_phase == "publish" else 1,
    )
    training_rows = [(dict(label="2", prompt=[dict(content="p")]), [1, 2])] * 4
    tokenizer = SimpleNamespace(eos_token_id=0, decode=lambda _: "\\boxed{2}")
    monkeypatch.setattr(
        beam.PreTrainedTokenizerFast, "from_pretrained", lambda *a, **kw: tokenizer
    )
    monkeypatch.setattr(
        beam, "load_data", lambda *a: (training_rows, training_rows[:1] * 30)
    )
    monkeypatch.setattr(beam.importlib.metadata, "version", lambda _: "test")
    calls = []

    def sample(sampler, item, cfg, tokenizer, replay, **kwargs):
        calls.append((sampler, replay))
        sequences = [
            beam.ReplaySequence(
                tokens=[2, 3],
                logprobs=[-1.0, -1.0],
                stop_reason="stop",
                replay=beam.spindle.replay.SamplingReplay(
                    prompt_tokens=2,
                    temperature=1.0,
                    routed_experts=base64.b64encode(bytes(3 * 52 * 8 * 4)).decode(),
                ),
            )
            for _ in range(2)
        ]
        return sequences, [1.0, 0.0]

    monkeypatch.setattr(beam, "sample_group", sample)

    class Future:
        def __init__(self, value):
            self.value = value

        def result(self, **kwargs):
            return self.value

    states, sampler_states, instances = {}, {}, []
    monkeypatch.setattr(beam, "plot", lambda *a, **kw: None)

    failures = [True] if failure_phase else []

    class Training:
        def __init__(self, state=None):
            self.step, self.weight, self.moment = state or (0, 1.0, 0.0)
            instances.append(self)

        def save_state(self, name, **kwargs):
            path = f"tinker://state/{name}"
            states[path] = (self.step, self.weight, self.moment)
            return Future(SimpleNamespace(path=path))

        def save_weights_for_sampler(self, name, **kwargs):
            sampler_states[name] = self.step
            return Future(SimpleNamespace(path=name))

        def forward_backward(self, datums, loss, **kwargs):
            if failure_phase == "backward" and failures:
                failures.pop()
                raise RuntimeError("simulated backward failure")
            assert loss == "cispo"
            return Future(
                SimpleNamespace(
                    metrics={"loss": 0.1},
                    loss_fn_outputs=[
                        {"logprobs": datum.loss_fn_inputs["logprobs"]}
                        for datum in datums
                    ],
                )
            )

        def optim_step(self, params):
            if failure_phase == "rejected" and failures:
                failures.pop()
                return Future(SimpleNamespace(metrics={"update_successful:mean": 0.0}))
            self.step += 1
            self.moment = 0.9 * self.moment + 1
            self.weight -= params.learning_rate * self.moment
            return Future(
                SimpleNamespace(
                    metrics={"update_successful:mean": 1.0, "grad_norm": 0.1}
                )
            )

    async def create(**kwargs):
        return Training()

    async def restore(path):
        return Training(states[path])

    async def create_sampler(model_path):
        if failure_phase == "publish" and sampler_states[model_path] == 1 and failures:
            failures.pop()
            raise RuntimeError("simulated publish failure")
        return sampler_states[model_path]

    service = SimpleNamespace(
        create_lora_training_client_async=create,
        create_training_client_from_state_with_optimizer_async=restore,
        create_sampling_client_async=create_sampler,
    )
    monkeypatch.setattr(beam.tinker, "ServiceClient", lambda **kw: service)
    monkeypatch.setenv("TINKER_BASE_URL", "https://example.invalid")
    monkeypatch.setenv("TINKER_API_KEY", "test")
    original_capture = beam.capture

    def capture(*a, **kw):
        if failure_phase == "prefetch" and a[2] == 2 and failures:
            failures.pop()
            raise RuntimeError("simulated prefetch failure")
        return original_capture(*a, **kw)

    monkeypatch.setattr(beam, "capture", capture)
    if failure_phase:
        with pytest.raises(
            RuntimeError,
            match="Optimizer update rejected"
            if failure_phase == "rejected"
            else f"simulated {failure_phase} failure",
        ):
            beam.run(args, config)
    else:
        beam.run(args, config)
    assert instances[0].step == (0 if failure_phase in {"backward", "rejected"} else 1)
    before = len(calls)
    args.resume = [
        r
        for r in beam.read_records(output / "metrics.jsonl")
        if r["event"] == "checkpoint"
    ][-1]["path"]
    if failure_phase is None and not lose_cache:
        reference = args.resume
        beam.run(args, config)
        assert len(calls) == before
        probe = beam.read_records(output / "metrics.jsonl")[-1]
        assert probe["event"] == "restore_roundtrip"
        assert probe["reference"] == reference
        assert states[probe["path"]] == states[reference]
        assert instances[-1].step == 1
        assert instances[-1] is not instances[0]
        args.resume = probe["path"]
    if lose_cache:
        shutil.rmtree(cache)
    args.stop_after = None
    beam.run(args, config)
    events = beam.read_records(output / "metrics.jsonl")
    assert [r["step"] for r in events if r["event"] == "update"] == [1, 2]
    extra_groups = (
        config.groups * (2 if failure_phase in {"backward", "rejected"} else 1)
        if lose_cache
        else config.groups
        if failure_phase == "prefetch"
        else 0
    )
    assert len(calls) == before + extra_groups
    assert instances[-1] is not instances[0]
    assert instances[-1].step == 2
    assert instances[-1].moment == pytest.approx(1.9)
    assert instances[-1].weight == pytest.approx(
        1 - beam.learning_rate(1) - 1.9 * beam.learning_rate(2)
    )
    if lose_cache or failure_phase == "prefetch":
        assert calls[before:] == [(0, True)] * extra_groups
    assert [r["behavior_policy_step"] for r in events if r["event"] == "cycle"] == [
        0
    ] * (1 if failure_phase == "prefetch" else 2)
    assert events[-1]["event"] == "completed"


def test_expired_run_does_not_create_remote_client(tmp_path, monkeypatch):
    output, cache = tmp_path / "results", tmp_path / "cache"
    output.mkdir()
    (output / "identity.json").write_text(json.dumps(dict(run_id="expired", started=0)))

    def unexpected(**kwargs):
        pytest.fail("Expired run attempted remote client creation")

    monkeypatch.setattr(beam.tinker, "ServiceClient", unexpected)
    args = SimpleNamespace(output=output, replay_cache=cache, resume=None)
    with pytest.raises(TimeoutError, match="time budget exhausted"):
        beam.run(args, beam.Config())


def test_cache_lock_excludes_different_output_writer(tmp_path):
    cache = tmp_path / "shared-cache"
    with beam.Journal(cache):
        args = SimpleNamespace(output=tmp_path / "other-output", replay_cache=cache)
        with pytest.raises(RuntimeError, match="Another reference-run writer"):
            beam.run(args, beam.Config())


def test_bootstrap_retry_preserves_identity_without_remote_creation(
    tmp_path, monkeypatch
):
    output = tmp_path / "results"
    provenance = tmp_path / "deployment.json"
    provenance.write_text("{}")
    args = SimpleNamespace(
        output=output,
        replay_cache=tmp_path / "cache",
        resume=None,
        deployment_provenance=provenance,
    )
    monkeypatch.setattr(
        beam.PreTrainedTokenizerFast, "from_pretrained", lambda *a, **kw: None
    )
    monkeypatch.setattr(beam.importlib.metadata, "version", lambda _: "test")
    monkeypatch.setattr(beam, "plot", lambda *a, **kw: None)

    def fail_data(*args):
        raise RuntimeError("dataset temporarily unavailable")

    monkeypatch.setattr(beam, "load_data", fail_data)

    def unexpected(**kwargs):
        pytest.fail("Dataset failure allocated a trainer")

    monkeypatch.setattr(beam.tinker, "ServiceClient", unexpected)
    for _ in range(2):
        with pytest.raises(RuntimeError, match="dataset temporarily unavailable"):
            beam.run(args, beam.Config())
    assert (
        len(
            [
                r
                for r in beam.read_records(output / "metrics.jsonl")
                if r["event"] == "segment"
            ]
        )
        == 2
    )
    assert len(list(output.glob("identity.json"))) == 1


def test_process_deadline_interrupts_blocking_startup():
    started = time.monotonic()
    code = beam.supervise(
        [sys.executable, "-c", "import time; time.sleep(30)"],
        0.1,
        termination_grace=0.1,
    )
    assert code == 124
    assert time.monotonic() - started < 5


@pytest.mark.parametrize("normal_exit", [False, True])
def test_supervisor_kills_descendant_even_after_leader_exit(tmp_path, normal_exit):
    pid_file = tmp_path / "descendant"
    descendant = (
        "import os,signal,time,pathlib;signal.signal(signal.SIGTERM,signal.SIG_IGN);"
        f"pathlib.Path({str(pid_file)!r}).write_text(str(os.getpid()));time.sleep(30)"
    )
    leader = (
        "import subprocess,sys,time,pathlib;"
        f"p=subprocess.Popen([sys.executable,'-c',{descendant!r}]);"
        f"path=pathlib.Path({str(pid_file)!r});"
        "\nwhile not path.exists(): time.sleep(.005)\n"
        + ("sys.exit(0)" if normal_exit else "time.sleep(30)")
    )
    code = beam.supervise([sys.executable, "-c", leader], 0.8, termination_grace=0.2)
    assert code == (0 if normal_exit else 124)
    pid = int(pid_file.read_text())
    for _ in range(100):
        status = Path(f"/proc/{pid}/stat")
        if not status.exists() or status.read_text().split()[2] == "Z":
            break
        time.sleep(0.01)
    else:
        pytest.fail("Supervisor left a descendant running")


def test_supervisor_interruption_uses_grace_not_remaining_budget(tmp_path):
    ready = tmp_path / "worker-pid"
    worker = (
        "import os,signal,time,pathlib;signal.signal(signal.SIGTERM,signal.SIG_IGN);"
        f"pathlib.Path({str(ready)!r}).write_text(str(os.getpid()));time.sleep(60)"
    )
    code = (
        f"import sys;sys.path.insert(0,{str(Path(beam.__file__).parent)!r});import beam;"
        f"beam.supervise([sys.executable,'-c',{worker!r}],3600,termination_grace=.1)"
    )
    supervisor = subprocess.Popen([sys.executable, "-c", code], start_new_session=True)
    worker_pid = None
    try:
        deadline = time.monotonic() + 20
        while not ready.exists() and time.monotonic() < deadline:
            time.sleep(0.02)
        assert ready.exists(), "Worker did not initialize"
        worker_pid = int(ready.read_text())
        started = time.monotonic()
        supervisor.send_signal(signal.SIGTERM)
        assert supervisor.wait(timeout=3) == 143
        assert time.monotonic() - started < 3
    finally:
        if worker_pid is not None:
            try:
                os.killpg(worker_pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        if supervisor.poll() is None:
            os.killpg(supervisor.pid, signal.SIGKILL)
            supervisor.wait(timeout=3)
