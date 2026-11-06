import asyncio

import httpx
import pytest

from spindle.engine import Engine, FutureStatus, OperationKind
from spindle.engine.api import Command
from spindle.engine.backend_http import HttpBackendClient, create_backend_app
from spindle.engine.operations import parse_operation_payload
from spindle.engine.spmd import DistributedExecutor
from tests.engine.test_server import forward, forward_backward
from tests.engine.test_spmd import RecordingBackend
from tests.support import EchoExecutor


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("failure", [False, True])
def test_overlap_preserves_sequence_and_drains_before_close(enabled, failure):
    async def run():
        blocked, release = asyncio.Event(), asyncio.Event()
        paired, finish = asyncio.Event(), asyncio.Event()
        calls = []

        class Executor(EchoExecutor):
            async def execute_forward_backward_batch(self, commands):
                calls.append(tuple(item.model_id for item in commands))
                if commands[0].model_id == "blocker":
                    blocked.set()
                    await release.wait()
                return await super().execute_forward_backward_batch(commands)

            async def execute_optimizer_forward_backward(self, commands):
                calls.append(tuple(item.model_id for item in commands))
                paired.set()
                await finish.wait()
                if failure:
                    raise RuntimeError("paired execution failed")
                return tuple(
                    [
                        await self.execute(item.model_id, item.kind, item.payload)
                        for item in commands
                    ]
                )

        server = Engine(Executor(), overlap_optimizer_step=enabled)
        for model in ("blocker", "a", "b"):
            await server.accept_model(model, {})
        await forward_backward(server, 1, model_id="blocker")
        await asyncio.wait_for(blocked.wait(), 1)
        optimizer = await server.optim_step(
            {"model_id": "a", "seq_id": 1, "adam_params": {}}
        )
        backward = await forward_backward(server, 1, model_id="b")
        next_a = await forward(server, 2, model_id="a")
        release.set()
        if enabled:
            await asyncio.wait_for(paired.wait(), 1)
            assert calls == [("blocker",), ("a", "b")]
            assert (
                await server.retrieve_future(optimizer)
            ).status == FutureStatus.PENDING
            assert (await server.retrieve_future(next_a)).status == FutureStatus.PENDING
            closing = asyncio.create_task(server.close())
            await asyncio.sleep(0)
            assert not closing.done()
            finish.set()
            await asyncio.wait_for(closing, 1)
            expected = FutureStatus.FAILED if failure else FutureStatus.COMPLETE
            assert (await server.retrieve_future(optimizer)).status == expected
            assert (await server.retrieve_future(backward)).status == expected
            assert calls.count(("a", "b")) == 1
        else:
            assert (
                await server.retrieve_future(next_a, timeout=1)
            ).status == FutureStatus.COMPLETE
            assert not paired.is_set()
            await server.close()

    asyncio.run(run())


def test_overlap_http_dispatch_and_adapter_exclusion():
    class Backend(RecordingBackend):
        def optim_step_with_forward_backward(self, model_id, adam, batch):
            (output,) = self.optim_step((model_id,), adam)
            return output, self.forward_backward(batch)

    async def run():
        backend = Backend()
        executor = DistributedExecutor(backend)
        client = HttpBackendClient(
            "http://backend",
            transport=httpx.ASGITransport(app=create_backend_app(executor)),
        )
        optimizer = Command(
            "a",
            OperationKind.OPTIM_STEP,
            parse_operation_payload(OperationKind.OPTIM_STEP, {"adam_params": {}}),
        )
        backward = Command(
            "b",
            OperationKind.FORWARD_BACKWARD,
            parse_operation_payload(
                OperationKind.FORWARD_BACKWARD, {"data": [], "loss_fn": "cross_entropy"}
            ),
        )
        results = await client.execute_optimizer_forward_backward((optimizer, backward))
        assert results[0] == {"metrics": {"grad_norm:mean": 0.5}}
        assert results[1]["metrics"] == {"loss:mean": 1.0}
        with pytest.raises(RuntimeError, match="different adapters"):
            await client.execute_optimizer_forward_backward(
                (optimizer, Command("a", backward.kind, backward.payload))
            )
        assert len(backend.calls) == 2
        await client.close()

    asyncio.run(run())
