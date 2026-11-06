import sys
from contextlib import nullcontext
from types import ModuleType, SimpleNamespace

import pytest
import torch

from spindle.backends.miles_runtime.optimizer_overlap import (
    detached_slot_state,
    optim_step_with_forward_backward,
)


@pytest.mark.parametrize("failure", [False, True])
def test_detached_gradients_survive_shared_buffer_reset_and_restore_views(failure):
    buffer = torch.tensor([2.0, 3.0, 5.0, 7.0])
    active = torch.nn.Parameter(torch.ones(2))
    other = torch.nn.Parameter(torch.ones(2))
    active.main_grad, other.main_grad = buffer[:2], buffer[2:]
    active.grad = torch.tensor([11.0, 13.0])
    original_main, original_grad = active.main_grad, active.grad
    copies = []
    with (
        pytest.raises(RuntimeError, match="optimizer failed")
        if failure
        else nullcontext()
    ):
        with detached_slot_state([active]) as copies:
            assert torch.equal(active.main_grad, torch.tensor([2.0, 3.0]))
            assert torch.equal(active.grad, torch.tensor([11.0, 13.0]))
            assert torch.equal(other.main_grad, torch.tensor([5.0, 7.0]))
            buffer.zero_()
            other.main_grad.add_(17)
            assert torch.equal(active.main_grad, torch.tensor([2.0, 3.0]))
            if failure:
                raise RuntimeError("optimizer failed")
    assert active.main_grad is original_main
    assert active.grad is original_grad
    assert torch.equal(other.main_grad, torch.tensor([17.0, 17.0]))
    assert torch.equal(copies.gradients[0], torch.tensor([2.0, 3.0]))
    assert torch.equal(copies.gradients[1], torch.tensor([11.0, 13.0]))


def test_detachment_restores_parameters_when_cloning_fails():
    parameter = torch.nn.Parameter(torch.ones(2))
    main = parameter.main_grad = torch.tensor([3.0, 4.0])
    broken = SimpleNamespace(main_grad=object(), grad=None, data=torch.ones(1))
    with pytest.raises(AttributeError):
        with detached_slot_state([parameter, broken]):
            pass
    assert parameter.main_grad is main


@pytest.mark.parametrize("forward_fails", [False, True])
def test_optimizer_drains_before_gather_and_does_not_use_backward_buffers(
    monkeypatch, forward_fails
):
    buffer = torch.tensor([2.0, 3.0])
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    parameter.main_grad = buffer[:1]
    trace, queued = [], []

    class Stream:
        def wait_stream(self, compute):
            trace.append("wait")

        def synchronize(self):
            trace.append("drain")
            for action in queued:
                action()

    stream = Stream()
    monkeypatch.setattr(torch.cuda, "Stream", lambda: stream)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda: "compute")
    monkeypatch.setattr(torch.cuda, "stream", lambda stream: nullcontext())
    module = ModuleType("miles.backends.megatron_utils.lora.optimizer")
    module.adapter_slot_parameters = lambda model, slot: [parameter]
    monkeypatch.setitem(sys.modules, module.__name__, module)

    class Optimizer:
        def apply_adam_params(self, params):
            pass

        def prepare_grads(self):
            self.gradient = parameter.main_grad

        def clip_and_step(self, norm):
            trace.append("enqueue")
            weight = parameter.data
            queued.append(lambda: weight.sub_(0.1 * self.gradient))
            return {"grad_norm": 2.0}

        def zero_grads(self):
            gradient = parameter.main_grad
            queued.append(lambda: gradient.zero_())

        def allgather_params(self):
            assert torch.equal(parameter, torch.tensor([0.8]))
            trace.append("gather")

    def forward(**kwargs):
        trace.append("forward")
        assert torch.equal(parameter, torch.tensor([1.0]))
        buffer.zero_()
        buffer[1] = 7
        if forward_fails:
            raise RuntimeError("forward failed")
        return {"per_datum": []}

    actor = SimpleNamespace(
        model=[], slot_optimizers={0: Optimizer()}, forward_backward=forward
    )
    with (
        pytest.raises(RuntimeError, match="forward failed")
        if forward_fails
        else nullcontext()
    ):
        result = optim_step_with_forward_backward(
            actor,
            batch_id=1,
            rollout_data_ref=None,
            adam_params_by_slot={0: {"grad_clip_norm": 1.0}},
            forward_only=False,
        )
        assert result["optimizer_outcomes"] == {0: {"grad_norm": 2.0}}
    assert trace == ["wait", "enqueue", "forward", "drain"] + (
        [] if forward_fails else ["gather"]
    )
    expected = 1.0 if forward_fails else 0.8
    assert torch.equal(parameter, torch.tensor([expected]))
    assert torch.equal(buffer, torch.tensor([0.0, 7.0]))
