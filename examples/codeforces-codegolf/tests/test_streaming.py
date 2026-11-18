import asyncio
import dataclasses
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from spindle.backends.megatron_runtime.common.forward_backward import (  # noqa: E402
    _loss,
)

from codegolf.config import Config, config_for, with_config_defaults  # noqa: E402
from codegolf.train import forward_backward, streamed_forward  # noqa: E402


def record(lengths):
    return {
        "prompt": [10, 11],
        "rows": [
            {
                "tokens": [20] * length,
                "logprobs": [-0.2] * length,
                "reward": float(index),
            }
            for index, length in enumerate(lengths)
        ],
    }


class GradientTrainer:
    def __init__(self):
        self.parameter = torch.tensor(0.03, dtype=torch.float64, requires_grad=True)
        self.finished = asyncio.Event()

    async def forward_backward_async(self, items, loss, *, loss_fn_config):
        objective = self.parameter * 0
        for item in items:
            inputs = item.loss_fn_inputs
            sampling = torch.tensor(inputs["logprobs"].data, dtype=torch.float64)
            target = sampling + self.parameter * torch.arange(len(sampling))
            loss_value, _ = _loss(
                target,
                {
                    "loss_name": loss,
                    "loss_config": loss_fn_config,
                    "sampling_logprobs": sampling,
                    "advantages": torch.tensor(inputs["advantages"].data),
                    "loss_mask": torch.tensor(inputs["weights"].data),
                },
            )
            objective += loss_value

        async def result_async():
            objective.backward()
            self.finished.set()
            return SimpleNamespace(metrics={"loss:sum": objective.item()})

        return SimpleNamespace(result_async=result_async)


@pytest.mark.parametrize("estimator", ["grpo", "tailrl"])
def test_streaming_overlaps_slow_group_and_preserves_accumulated_gradient(estimator):
    async def run():
        cfg = dataclasses.replace(
            config_for("stream-v11"),
            prompts_per_step=2,
            group_size=3,
            advantage_estimator=estimator,
        )
        records = [record([1, 3, 8]), record([7, 2, 5])]
        streamed, batched = GradientTrainer(), GradientTrainer()

        async def ready():
            return records[0]

        async def slow():
            await asyncio.wait_for(streamed.finished.wait(), 1)
            return records[1]

        completed, metrics, timing = await streamed_forward(
            streamed, [slow(), ready()], cfg
        )
        await forward_backward(
            batched, records, dataclasses.replace(cfg, stream_minibatches=False)
        )
        assert completed == records
        assert len(metrics["minibatches"]) == timing["minibatches"] == 2
        assert streamed.parameter.grad.item() == pytest.approx(
            batched.parameter.grad.item()
        )
        assert abs(streamed.parameter.grad.item()) > 0.01

    asyncio.run(run())


@pytest.mark.parametrize("failure", ["rollout", "forward", "cancel"])
def test_partial_stream_failure_drains_pending_rollouts(failure):
    async def run():
        trainer = GradientTrainer()
        waiting, cleaned = asyncio.Event(), asyncio.Event()
        cfg = dataclasses.replace(config_for("stream-v11"), group_size=3)
        records = record([1, 3, 8])

        async def ready():
            await waiting.wait()
            return records

        async def pending():
            waiting.set()
            try:
                await trainer.finished.wait()
                if failure == "rollout":
                    raise RuntimeError("rollout failed after partial accumulation")
                await asyncio.Event().wait()
            finally:
                await asyncio.sleep(0)
                cleaned.set()

        if failure == "forward":
            original = trainer.forward_backward_async

            async def fail(*args, **kwargs):
                future = await original(*args, **kwargs)
                await future.result_async()
                raise RuntimeError("forward response lost after accumulation")

            trainer.forward_backward_async = fail
        task = asyncio.create_task(streamed_forward(trainer, [ready(), pending()], cfg))
        if failure == "cancel":
            await asyncio.wait_for(trainer.finished.wait(), 1)
            task.cancel()
        with pytest.raises(
            asyncio.CancelledError if failure == "cancel" else ExceptionGroup
        ):
            await asyncio.wait_for(task, 1)
        assert trainer.parameter.grad is not None
        assert cleaned.is_set()

    asyncio.run(run())


def test_streaming_configs_and_legacy_resume():
    streamed = dataclasses.asdict(config_for("stream-v11"))
    batched = dataclasses.asdict(config_for("batch-v11"))
    assert streamed == {**batched, "stream_minibatches": True}
    legacy = dataclasses.asdict(Config())
    del legacy["stream_minibatches"], legacy["sequence_mean"]
    assert with_config_defaults(legacy) == dataclasses.asdict(Config())
    with pytest.raises(ValueError, match="Streaming requires"):
        Config(stream_minibatches=True)
