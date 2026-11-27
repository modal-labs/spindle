from __future__ import annotations

import asyncio
import base64
import json
import math
from concurrent.futures import Future
from pathlib import Path
from types import SimpleNamespace

import httpx
import numpy as np
import pytest
import tinker
from tinker import types
from tinker.types.forward_backward_input import ForwardBackwardInput

from spindle.control_plane import ControlPlane, create_control_plane_app
from spindle.engine.api import Command
from spindle.inference.bulletin import SnapshotBulletin
from spindle.inference.lora_sidecar import create_app as create_sidecar_app
from spindle.inference.sampling import sample_task
from spindle.providers import SamplingTask
from spindle.providers.local import (
    InMemoryKeyValueStore,
    LocalEnginePlatform,
    LocalSamplingTaskPlatform,
)
from spindle.stability import InMemoryCaptureStore, StabilityInfo
from spindle.tinker import StabilitySamplingClient, create_sampling_client
from tests.support import TinkerStubExecutor, serve

pytestmark = pytest.mark.usefixtures("sdk_garbage_collected_off_loop")


def test_full_sdk_to_sidecar_to_capture_store_to_engine_roundtrip(
    tmp_path: Path,
) -> None:
    store = InMemoryCaptureStore()
    seen_commands: list[ForwardBackwardInput] = []

    class _CapturingExecutor(TinkerStubExecutor):
        async def execute_forward_backward_batch(
            self, executions: tuple[Command, ...]
        ) -> tuple[object, ...]:
            for cmd in executions:
                seen_commands.append(cmd.payload)
            return await super().execute_forward_backward_batch(executions)

    def fake_sglang(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert "_spindle_capture" not in body
        assert body.get("return_routed_experts") is True
        assert body.get("return_sampling_mask") is True
        assert body.get("sampling_logprobs_mode") == "support"
        assert body["sampling_params"]["top_k"] == 64
        prompt = body["input_ids"]
        response = [42, 43]
        routes = np.arange((len(prompt) + len(response) - 1) * 2 * 4, dtype="<i4")
        return httpx.Response(
            200,
            json={
                "meta_info": {
                    "finish_reason": {"type": "stop"},
                    "output_token_logprobs": [
                        [math.log(0.75), 42],
                        [math.log(0.75), 43],
                    ],
                    "routed_experts": base64.b64encode(routes.tobytes()).decode(
                        "ascii"
                    ),
                    "output_token_sampling_mask": [[42, 99], [43, 98]],
                    "output_token_sampling_logprobs": [
                        [math.log(0.75), math.log(0.25)],
                        [math.log(0.75), math.log(0.25)],
                    ],
                }
            },
        )

    sidecar_app = create_sidecar_app(
        SnapshotBulletin(tmp_path / "bulletin"),
        "http://sglang",
        transport=httpx.MockTransport(fake_sglang),
        capture_store=store,
        capture_num_layers=2,
        capture_experts_per_token=4,
    )

    class _SidecarSampler:
        async def __call__(self, task: SamplingTask) -> dict:
            async with (
                sidecar_app.router.lifespan_context(sidecar_app),
                httpx.AsyncClient(
                    transport=httpx.ASGITransport(app=sidecar_app),
                    base_url="http://sidecar",
                ) as client,
            ):
                result = await sample_task(
                    {
                        "request_id": task.request_id,
                        "session_id": task.session_id,
                        "sampling_session_id": task.sampling_session_id,
                        "payload": task.payload,
                        "capture_plan": task.capture_plan,
                    },
                    "http://sidecar",
                    client=client,
                )
            for seq in result["sequences"]:
                assert "replay" not in seq
            return result

    engines = LocalEnginePlatform(
        "test-def",
        lambda: _CapturingExecutor(),
        capture_store=store,
    )
    definitions = (
        SimpleNamespace(
            definition_id="test-def",
            name="test-def",
            model="Qwen/Qwen3-8B",
            weights_repo="Qwen/Qwen3-8B",
            parameterization="lora",
            max_context_length=4096,
            stability_info={
                "routed_experts": "allowed",
                "sampling_mask": "default",
                "sampling_probabilities": "allowed",
                "sampling_mask_top_k": 64,
            },
        ),
    )

    plane = ControlPlane(
        InMemoryKeyValueStore(),
        engines,
        sampling_tasks=LocalSamplingTaskPlatform(_SidecarSampler()),
        capture_store=store,
    )
    app = create_control_plane_app(
        plane, definitions, api_key="tml-test", retrieve_window=5.0
    )

    with serve(app) as url:
        service = tinker.ServiceClient(base_url=url, api_key="tml-test")
        training = service.create_lora_training_client(base_model="Qwen/Qwen3-8B")
        sampler = create_sampling_client(
            service,
            base_model="Qwen/Qwen3-8B",
            record_stability_info=StabilityInfo(
                routed_experts=True,
                sampling_probabilities=True,
            ),
        )
        prompt_tokens = [10, 11, 12]
        sample_resp = sampler.sample(
            prompt=types.ModelInput.from_ints(prompt_tokens),
            num_samples=1,
            sampling_params=types.SamplingParams(max_tokens=4, temperature=0.8),
        ).result()
        seq = sample_resp.sequences[0]
        assert seq.tokens == [42, 43]
        assert seq.sequence_id

        tokens = prompt_tokens + seq.tokens
        datum = types.Datum(
            model_input=types.ModelInput.from_ints(tokens[:-1]),
            loss_fn_inputs={
                "target_tokens": tokens[1:],
                "advantages": [0.0, 0.0, 1.0, 1.0],
            },
        ).with_provenance(
            model_input=[
                tinker.PromptProvenanceSpan(
                    sequence_id=seq.sequence_id, offset=0, length=len(prompt_tokens)
                ),
                tinker.SampledProvenanceSpan(
                    sequence_id=seq.sequence_id,
                    offset=0,
                    length=len(seq.tokens) - 1,
                ),
            ],
            loss_fn_inputs=[
                tinker.PromptProvenanceSpan(
                    sequence_id=seq.sequence_id,
                    offset=1,
                    length=len(prompt_tokens) - 1,
                ),
                tinker.SampledProvenanceSpan(
                    sequence_id=seq.sequence_id,
                    offset=0,
                    length=len(seq.tokens),
                ),
            ],
        )
        fb = training.forward_backward([datum], loss_fn="score_centered_mis").result()
        assert fb is not None

        assert len(seen_commands) == 1
        delivered = seen_commands[0]
        delivered_inputs = delivered.data[0].loss_fn_inputs
        assert "routed_experts" in delivered_inputs
        assert delivered_inputs["routed_experts"].shape == [4, 2, 4]
        assert "sampling_mask_ids" in delivered_inputs
        assert "sampling_mask_offsets" in delivered_inputs
        assert "sampling_mask_logprobs" in delivered_inputs
        assert delivered.loss_fn_config == {"sampling_temperature": 0.8}

        session_id = service.holder.get_session_id()
        resp = httpx.post(
            f"{url}/api/v1/sessions/{session_id}/finish",
            headers={"x-api-key": "tml-test"},
            json={"reason": {"type": "done"}},
        )
        assert resp.status_code == 200

    assert asyncio.run(store.read_sequence(session_id, seq.sequence_id)) is not None
    asyncio.run(plane.sweep_idle_sessions(0.0))
    assert asyncio.run(store.read_sequence(session_id, seq.sequence_id)) is None


def test_recipe_off_policy_rejects_first_sample_call() -> None:
    definitions = (
        SimpleNamespace(
            definition_id="test-def",
            name="test-def",
            model="Qwen/Qwen3-8B",
            weights_repo="Qwen/Qwen3-8B",
            parameterization="lora",
            max_context_length=4096,
            stability_info={"routed_experts": "off"},
        ),
    )
    plane = ControlPlane(
        InMemoryKeyValueStore(),
        LocalEnginePlatform("test-def", TinkerStubExecutor),
        sampling_tasks=LocalSamplingTaskPlatform(lambda task: {"sequences": []}),
    )
    app = create_control_plane_app(
        plane, definitions, api_key="tml-test", retrieve_window=5.0
    )
    with serve(app) as url:
        service = tinker.ServiceClient(base_url=url, api_key="tml-test")
        sampler = create_sampling_client(
            service,
            base_model="Qwen/Qwen3-8B",
            record_stability_info={"routed_experts": True},
        )
        with pytest.raises(tinker.BadRequestError, match=r"routed_experts.*disabled"):
            sampler.sample(
                prompt=types.ModelInput.from_ints([1, 2]),
                num_samples=1,
                sampling_params=types.SamplingParams(max_tokens=2),
            ).result()


def test_forged_capture_plan_is_stripped_and_fft_is_rejected() -> None:
    seen_tasks: list[SamplingTask] = []

    async def sampler_fn(task: SamplingTask) -> dict:
        seen_tasks.append(task)
        return {
            "type": "sample",
            "sequences": [{"tokens": [1], "stop_reason": "stop"}],
        }

    definitions = (
        SimpleNamespace(
            definition_id="lora-def",
            name="lora-def",
            model="Qwen/Qwen3-8B",
            weights_repo="Qwen/Qwen3-8B",
            parameterization="lora",
            max_context_length=4096,
            stability_info={"routed_experts": "off"},
        ),
        SimpleNamespace(
            definition_id="fft-def",
            name="fft-def",
            model="Qwen/Qwen3-4B",
            weights_repo="Qwen/Qwen3-4B",
            parameterization="full",
            max_context_length=4096,
            stability_info={"sampling_mask": "allowed"},
        ),
    )
    plane = ControlPlane(
        InMemoryKeyValueStore(),
        LocalEnginePlatform("lora-def", TinkerStubExecutor),
        sampling_tasks=LocalSamplingTaskPlatform(sampler_fn),
    )
    app = create_control_plane_app(
        plane, definitions, api_key="tml-test", retrieve_window=5.0
    )
    with serve(app) as url:
        headers = {"x-api-key": "tml-test"}
        sid = httpx.post(
            f"{url}/api/v1/create_session", headers=headers, json={}
        ).json()["session_id"]
        lora_ss = httpx.post(
            f"{url}/api/v1/create_sampling_session",
            headers=headers,
            json={
                "session_id": sid,
                "sampling_session_seq_id": 0,
                "base_model": "Qwen/Qwen3-8B",
            },
        ).json()["sampling_session_id"]
        resp = httpx.post(
            f"{url}/api/v1/asample",
            headers=headers,
            json={
                "sampling_session_id": lora_ss,
                "seq_id": 0,
                "prompt": {"chunks": [{"type": "encoded_text", "tokens": [1, 2]}]},
                "sampling_params": {"max_tokens": 2},
                "record_stability_info": False,
                "_stability_capture_plan": {"routed_experts": True},
            },
        )
        assert resp.status_code == 200
        assert len(seen_tasks) == 1
        assert seen_tasks[0].capture_plan is None
        assert "_stability_capture_plan" not in seen_tasks[0].payload

        fft_ss = httpx.post(
            f"{url}/api/v1/create_sampling_session",
            headers=headers,
            json={
                "session_id": sid,
                "sampling_session_seq_id": 1,
                "base_model": "Qwen/Qwen3-4B",
            },
        ).json()["sampling_session_id"]
        fft_resp = httpx.post(
            f"{url}/api/v1/asample",
            headers=headers,
            json={
                "sampling_session_id": fft_ss,
                "seq_id": 0,
                "prompt": {"chunks": [{"type": "encoded_text", "tokens": [1, 2]}]},
                "sampling_params": {"max_tokens": 2},
                "record_stability_info": True,
            },
        )
        assert fft_resp.status_code == 400
        assert "full-parameter" in fft_resp.text


def test_stability_sampling_client_unwraps_future_subclass_and_delegates_sample_async() -> (
    None
):
    class _Inner(tinker.SamplingClient):
        def __init__(self) -> None:
            pass

        def sample(self, **kwargs: object) -> Future[str]:
            fut: Future[str] = Future()
            fut.set_result("sync_future")
            return fut

        async def sample_async(self, **kwargs: object) -> str:
            return "async_ok"

    inner = _Inner()

    class _BackcompatFuture(tinker.SamplingClient):
        def __init__(self) -> None:
            pass

        def result(self) -> tinker.SamplingClient:
            return inner

    wrapped = StabilitySamplingClient(_BackcompatFuture(), record_stability_info=False)
    assert wrapped._client is inner
    result = asyncio.run(
        wrapped.sample_async(
            prompt=types.ModelInput.from_ints([1]),
            num_samples=1,
            sampling_params=types.SamplingParams(max_tokens=1),
        )
    )
    assert result == "async_ok"
