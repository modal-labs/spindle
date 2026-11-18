import asyncio
import importlib.util
import json
import sys
from collections.abc import AsyncIterator
from contextlib import nullcontext
from pathlib import Path
from types import ModuleType, SimpleNamespace

import httpx
import pytest
import torch

from spindle.inference import lora_packing


@pytest.fixture
def loading_benchmark(
    monkeypatch: pytest.MonkeyPatch,
) -> ModuleType:
    monkeypatch.setenv("SPINDLE_LORA_BENCH_ASSET_VOLUME", "test-assets")
    monkeypatch.setenv("SPINDLE_LORA_BENCH_ADAPTER_VOLUME", "test-adapters")
    path = Path(__file__).parents[2] / "scripts/benchmark_lora_loading.py"
    spec = importlib.util.spec_from_file_location("loading_stream_test", path)
    assert spec is not None and spec.loader is not None
    benchmark = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, benchmark)
    spec.loader.exec_module(benchmark)
    return benchmark


def test_stream_framing_preserves_unicode_line_separators(
    loading_benchmark: ModuleType,
) -> None:
    payload = {"text": "a\u2028b\u2029c\u0085d", "meta_info": {"completion_tokens": 1}}
    message = "data: " + json.dumps(payload, ensure_ascii=False)

    class FragmentedStream(httpx.AsyncByteStream):
        async def __aiter__(self) -> AsyncIterator[bytes]:
            for value in (message + "\r\n\r\ndata: [DONE]\n\n").encode():
                yield bytes([value])

    async def read() -> list[str]:
        response = httpx.Response(200, stream=FragmentedStream())
        try:
            return [line async for line in loading_benchmark.sse_lines(response)]
        finally:
            await response.aclose()

    lines = asyncio.run(read())
    assert lines == [message, "", "data: [DONE]", ""]
    assert json.loads(lines[0][6:]) == payload


@pytest.mark.parametrize("fail_load", [False, True])
def test_periodic_syncs_use_fresh_versions_and_join_all_requests(
    loading_benchmark: ModuleType, fail_load: bool
) -> None:
    async def run() -> None:
        loaded = []
        completed: dict[str, int] = {}
        active = set()
        stream_started = asyncio.Event()
        blocked = asyncio.Event()

        async def register(name: str) -> dict:
            if fail_load and name == "test-client1-version0":
                await stream_started.wait()
                raise RuntimeError("registration failed")
            client, version = name.removeprefix("test-client").split("-version")
            if int(version):
                assert completed[f"test-client{client}-version{int(version) - 1}"] == 2
            loaded.append(name)
            await asyncio.sleep(0)
            return {"name": name, "start": 0.0, "end": 0.1}

        class Samples(httpx.AsyncByteStream):
            def __init__(self, name: str) -> None:
                self.name = name

            async def __aiter__(self) -> AsyncIterator[bytes]:
                active.add(self)
                stream_started.set()
                try:
                    if fail_load:
                        await blocked.wait()
                    yield b'data: {"meta_info": {"completion_tokens": 128}}\n\n'
                    completed[self.name] = completed.get(self.name, 0) + 1
                finally:
                    active.remove(self)

        def generate(request: httpx.Request) -> httpx.Response:
            name = json.loads(request.content)["lora_path"]
            assert name in loaded
            return httpx.Response(200, stream=Samples(name))

        steps = []
        async with httpx.AsyncClient(
            base_url="http://test", transport=httpx.MockTransport(generate)
        ) as client:
            pending = loading_benchmark.periodic_sync_traffic(
                client, "test", register, steps.append, clients=2, rounds=3
            )
            if fail_load:
                with pytest.raises(ExceptionGroup, match="TaskGroup") as failure:
                    await pending
                assert "registration failed" in str(failure.value.exceptions[0])
                assert not steps
            else:
                result = await pending
                assert result["loads"] == len(set(loaded)) == len(steps) == 6
                assert result["output_tokens"] == 2 * 3 * 2 * 128
                assert set(completed.values()) == {2}
                assert result["output_tps_including_loads"] > 0
        assert not active

    asyncio.run(run())


@pytest.mark.parametrize("installed", [False, True])
def test_packing_does_not_mutate_active_slots_and_upload_owns_cpu_data(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    installed: bool,
) -> None:
    path = Path(__file__).parents[2] / "scripts/lora_loading_probe.py"
    spec = importlib.util.spec_from_file_location("loading_probe_test", path)
    assert spec is not None and spec.loader is not None
    probe = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(probe)
    monkeypatch.setenv("SPINDLE_LORA_TRACE_ROOT", str(tmp_path))
    zeros = torch.zeros

    def cpu_zeros(*args: object, **kwargs: object) -> torch.Tensor:
        kwargs.pop("pin_memory", None)
        return zeros(*args, **kwargs)

    monkeypatch.setattr(torch, "zeros", cpu_zeros)
    monkeypatch.setattr(torch.cuda, "device", lambda device: nullcontext())

    class Pool:
        pin_memory_available = True

        def __init__(self) -> None:
            self.A_buffer = {"layer": [torch.full((2, 4, 3), 37.0)]}
            self.B_buffer = {"layer": [torch.full((2, 5, 4), 37.0)]}
            self.embedding_A_buffer = {}
            self.embedding_B_buffer = {}
            self.lm_head_A_buffer = {}
            self.lm_head_B_buffer = {}
            self.new_embeddings_buffer = {}

        def retain_local_expert_weights(self, adapter: SimpleNamespace) -> None:
            pass

        def load_lora_weight_to_buffer(
            self,
            uid: str,
            buffer_id: int,
            lora_adapter: SimpleNamespace,
            lora_modules: object,
            lora_embed_tokens_module: object,
            lora_lm_head_module: object,
        ) -> None:
            self.A_buffer["layer"][0][buffer_id].zero_()
            self.B_buffer["layer"][0][buffer_id].zero_()
            self.A_buffer["layer"][0][buffer_id, :2].copy_(lora_adapter.a)
            self.B_buffer["layer"][0][buffer_id, :, :2].copy_(lora_adapter.b)

    class Manager(SimpleNamespace):
        def prepare_lora_adapter(
            self, lora_ref: object
        ) -> tuple[None, SimpleNamespace]:
            return None, self.adapter

    if installed:
        monkeypatch.setattr(lora_packing, "install_readiness", lambda: None)
        monkeypatch.setattr(lora_packing, "install_admission", lambda: None)
        monkeypatch.setitem(
            sys.modules,
            "sglang.srt.lora.lora_manager",
            SimpleNamespace(LoRAManager=Manager),
        )
        monkeypatch.setitem(
            sys.modules,
            "sglang.srt.lora.mem_pool",
            SimpleNamespace(LoRAMemoryPool=Pool),
        )
        lora_packing.install()
        lora_packing.install()
    else:
        probe.patch_pool(
            Pool,
            {
                "copy_weight_into_buffer": lambda destination, source: (
                    destination.copy_(source)
                )
            },
        )

    pool = Pool()
    adapter = SimpleNamespace(
        uid="test",
        a=torch.arange(6.0).reshape(2, 3),
        b=torch.arange(10.0).reshape(5, 2),
    )
    manager = Manager(
        adapter=adapter,
        memory_pool=pool,
        device="cpu",
        lora_modules=[],
        embed_tokens_module=None,
        lm_head_module=None,
    )
    expected_a, expected_b = adapter.a.clone(), adapter.b.clone()
    if installed:
        manager.prepare_lora_adapter("test")
    else:
        probe._pack(manager, adapter)
    assert torch.all(pool.A_buffer["layer"][0] == 37)
    assert torch.all(pool.B_buffer["layer"][0] == 37)
    adapter.a.fill_(-1)
    adapter.b.fill_(-1)
    pool.load_lora_weight_to_buffer("test", 1, adapter, [], None, None)
    torch.testing.assert_close(pool.A_buffer["layer"][0][1, :2], expected_a)
    torch.testing.assert_close(pool.B_buffer["layer"][0][1, :, :2], expected_b)
    assert not pool.A_buffer["layer"][0][1, 2:].count_nonzero()
    assert not pool.B_buffer["layer"][0][1, :, 2:].count_nonzero()
    assert torch.all(pool.A_buffer["layer"][0][0] == 37)
    assert torch.all(pool.B_buffer["layer"][0][0] == 37)
    replacement = Pool()
    assert not adapter._spindle_upload_plan.copy_to(replacement, 0)
    assert torch.all(replacement.A_buffer["layer"][0] == 37)
    assert torch.all(replacement.B_buffer["layer"][0] == 37)
    replacement.load_lora_weight_to_buffer(
        uid="test",
        buffer_id=0,
        lora_adapter=adapter,
        lora_modules=[],
        lora_embed_tokens_module=None,
        lora_lm_head_module=None,
    )
    assert torch.all(replacement.A_buffer["layer"][0][0, :2] == -1)
    assert torch.all(replacement.B_buffer["layer"][0][0, :, :2] == -1)
    pool.B_buffer["layer"][0] = torch.full((2, 6, 4), 47.0)
    assert not adapter._spindle_upload_plan.copy_to(pool, 0)
    assert torch.all(pool.A_buffer["layer"][0][0] == 37)
    assert torch.all(pool.B_buffer["layer"][0] == 47)
    before_a = pool.A_buffer["layer"][0].clone()
    before_b = pool.B_buffer["layer"][0].clone()
    invalid = SimpleNamespace(uid="invalid", a=adapter.a, b=torch.zeros(8, 2))
    with pytest.raises(RuntimeError, match="size"):
        lora_packing.build_upload_plan(
            manager, invalid, Pool.load_lora_weight_to_buffer
        )
    assert torch.equal(pool.A_buffer["layer"][0], before_a)
    assert torch.equal(pool.B_buffer["layer"][0], before_b)
