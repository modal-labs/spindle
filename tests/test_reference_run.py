import dataclasses
import json
from collections import Counter

import pytest
import tinker
from tinker import AdamParams
from tinker.types.forward_backward_input import ForwardBackwardInput

from reference_runs import dapo_math
from spindle.control_plane import ControlPlane, create_control_plane_app
from spindle.engine import OperationKind
from spindle.providers.local import (
    InMemoryKeyValueStore,
    LocalEnginePlatform,
    LocalSamplingTaskPlatform,
)
from tests.support import TinkerStubExecutor, serve

pytestmark = pytest.mark.usefixtures("sdk_garbage_collected_off_loop")

BASE_MODEL = "Qwen/Qwen3-8B"
CONFIG = dapo_math.RunConfig(
    **{**dapo_math.SMOKE, "model": BASE_MODEL, "max_response_tokens": 4}
)
ROWS = [
    {"prompt": [{"role": "user", "content": str(index)}], "label": "0"}
    for index in range(100)
]


class Tokenizer:
    eos_token_id = 1

    def convert_tokens_to_ids(self, token):
        return 2

    def apply_chat_template(self, messages, **_):
        return [1000 + int(messages[0]["content"])] * 3

    def decode(self, tokens):
        return f"\\boxed{{{tokens[0] % 2}}}"


class Server:
    """What the stub service saw, and the optimizer outcomes it should fake."""

    def __init__(self):
        self.optimizer_requests = Counter()
        self.optimizer_faults = {}
        self.sampled = []
        self.training_clients = 0


@pytest.fixture
def server():
    state = Server()

    class Executor(TinkerStubExecutor):
        async def accept_model(self, model_id, spec):
            state.training_clients += 1

        async def execute(self, model_id, kind, payload):
            if kind == OperationKind.FORWARD_BACKWARD:
                assert isinstance(payload, ForwardBackwardInput)
                return {
                    "loss_fn_output_type": "scalar",
                    "loss_fn_outputs": [
                        {
                            "logprobs": {
                                "data": [-0.25] * datum.model_input.length,
                                "dtype": "float32",
                                "shape": [datum.model_input.length],
                            }
                        }
                        for datum in payload.data
                    ],
                    "metrics": {"loss:sum": 1.25},
                }
            if kind == OperationKind.OPTIM_STEP:
                assert isinstance(payload, AdamParams)
                state.optimizer_requests["total"] += 1
                fault = state.optimizer_faults.pop(
                    state.optimizer_requests["total"], None
                )
                if fault == "lost":
                    raise RuntimeError("the reply was lost")
                if fault == "rejected":
                    return {"metrics": {"update_successful:mean": 0.0}}
                return {
                    "metrics": {"update_successful:mean": 1.0, "grad_norm:mean": 0.5}
                }
            return await super().execute(model_id, kind, payload)

    async def sampler(task):
        tokens = task.payload["prompt"]["chunks"][0]["tokens"]
        state.sampled.append((task.model_path, tokens[0] - 1000))
        return {
            "type": "sample",
            "sequences": [
                {
                    "stop_reason": "length",
                    "tokens": list(range(index, index + 4)),
                    "logprobs": [-0.5] * 4,
                }
                for index in range(task.payload["num_samples"])
            ],
        }

    async def checkpoint_metadata(path):
        return {
            "schema_version": 1,
            "base_model": BASE_MODEL,
            "engine_definition_id": "qwen3_8b",
            "parameterization": {"type": "lora"},
            "lora_config": {
                "rank": CONFIG.lora_rank,
                "train_mlp": True,
                "train_attn": True,
                "train_unembed": True,
            },
        }

    plane = ControlPlane(
        InMemoryKeyValueStore(),
        LocalEnginePlatform("qwen3_8b", Executor),
        sampling_tasks=LocalSamplingTaskPlatform(sampler),
        read_checkpoint_metadata=checkpoint_metadata,
    )
    definition = dataclasses.make_dataclass(
        "Definition",
        ["definition_id", "name", "model", "parameterization", "max_context_length"],
    )("qwen3_8b", "qwen3_8b", BASE_MODEL, "lora", 32768)
    app = create_control_plane_app(
        plane, (definition,), api_key="tml-test", retrieve_window=5.0
    )
    with serve(app) as url:
        state.url = url
        yield state


def run(server, output, **options):
    """One client process: a new SDK connection with no in-memory training state."""
    dapo_math.run(
        CONFIG,
        output,
        service=tinker.ServiceClient(base_url=server.url, api_key="tml-test"),
        tokenizer=Tokenizer(),
        rows=ROWS,
        **options,
    )


def events(output, event):
    return [
        r
        for r in dapo_math.read_records(output / "metrics.jsonl")
        if r["event"] == event
    ]


def training_ids(config, update):
    _, training = dapo_math.select(config, ROWS, Tokenizer())
    start = (update - 1) * config.groups
    return {p["id"] for p in training[start : start + config.groups]}


def test_stop_and_resume_keeps_cursor_and_behavior_policy(server, tmp_path):
    run(server, tmp_path, stop_after=2)
    checkpoint = events(tmp_path, "checkpoint")[-1]
    assert checkpoint["step"] == 2 and checkpoint["next_cursor"] == 2 * CONFIG.groups
    assert events(tmp_path, "stopped")
    before = len(server.sampled)

    run(server, tmp_path, resume=True)

    assert server.training_clients == 2
    assert events(tmp_path, "restore")[0]["path"] == checkpoint["path"]
    assert [u["step"] for u in events(tmp_path, "update")] == [1, 2, 3]
    third = events(tmp_path, "update")[-1]
    assert third["cursor"] == checkpoint["next_cursor"]
    assert third["sampler_path"] == checkpoint["sampler_path"]
    assert third["behavior_policy_step"] == 2
    assert third["grad_norm"] == 0.5
    assert third["sampled_log_ratio_mean"] == pytest.approx(0.25)
    assert "seconds_since_previous_update" not in third
    assert {
        row for path, row in server.sampled[before : before + CONFIG.groups]
    } == training_ids(CONFIG, 3)
    assert {path for path, _ in server.sampled[before : before + CONFIG.groups]} == {
        checkpoint["sampler_path"]
    }
    evaluated = {e["step"]: e for e in events(tmp_path, "evaluation")}
    assert sorted(evaluated) == [0, 1, 2, 3]
    assert evaluated[3]["sampler_path"] == events(tmp_path, "publication")[-1]["path"]
    assert set(evaluated[0]["correct_per_problem"]) == {1}
    assert not set(events(tmp_path, "segment")[0]["evaluation_ids"]) & {
        row for _, row in server.sampled[before : before + CONFIG.groups]
    }
    assert events(tmp_path, "completed")


@pytest.mark.parametrize(
    ("fault", "outcome", "resumable"),
    [("lost", "unknown", False), ("rejected", "rejected", True)],
)
def test_unaccepted_optimizer_request_is_never_sent_again(
    server, tmp_path, fault, outcome, resumable
):
    server.optimizer_faults[2] = fault
    with pytest.raises(SystemExit, match=outcome):
        run(server, tmp_path)
    records = dapo_math.read_records(tmp_path / "metrics.jsonl")
    assert dapo_math.optimizer_outcomes(records) == {1: "accepted", 2: outcome}
    assert [u["step"] for u in events(tmp_path, "update")] == [1]
    assert server.optimizer_requests["total"] == 2

    if not resumable:
        with pytest.raises(SystemExit, match="Start a new run"):
            run(server, tmp_path, resume=True)
        assert server.training_clients == 1
        assert server.optimizer_requests["total"] == 2
        return

    sampled = len(server.sampled)
    run(server, tmp_path, resume=True)
    assert [u["step"] for u in events(tmp_path, "update")] == [1, 2, 3]
    assert len(events(tmp_path, "rollout")) == 3
    assert len(server.sampled) - sampled == CONFIG.groups + 2 * CONFIG.eval_problems


def test_missing_pending_batch_stops_the_resume(server, tmp_path):
    server.optimizer_faults[2] = "rejected"
    with pytest.raises(SystemExit):
        run(server, tmp_path)
    (tmp_path / "local" / "rollout-0002.json").unlink()

    with pytest.raises(SystemExit, match="missing or changed"):
        run(server, tmp_path, resume=True)
    assert server.optimizer_requests["total"] == 2
    assert len(events(tmp_path, "rollout")) == 2


def test_interrupted_write_keeps_earlier_records(tmp_path):
    with dapo_math.Journal(tmp_path) as journal:
        journal.event("update", step=1)
        with pytest.raises(SystemExit, match="Another process"):
            dapo_math.Journal(tmp_path)
    metrics = tmp_path / "metrics.jsonl"
    with metrics.open("ab") as stream:
        stream.write(b'{"event": "update", "st')
    assert [r["step"] for r in dapo_math.read_records(metrics)] == [1]

    with dapo_math.Journal(tmp_path) as journal:
        journal.event("update", step=2)
    assert [r["step"] for r in dapo_math.read_records(metrics)] == [1, 2]


def test_existing_run_is_not_overwritten(server, tmp_path):
    run(server, tmp_path, stop_after=0)
    before = (tmp_path / "metrics.jsonl").read_bytes()
    with pytest.raises(SystemExit, match="--resume"):
        run(server, tmp_path)
    assert (tmp_path / "metrics.jsonl").read_bytes() == before
    assert json.loads((tmp_path / "config.json").read_text())["config"]["groups"] == (
        CONFIG.groups
    )
