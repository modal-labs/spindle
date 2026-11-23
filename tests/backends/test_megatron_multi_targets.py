"""Candidate scoring through FFT packing, loss gradients, and output reconstruction."""

import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from tinker import Datum, ModelInput, TensorData

from spindle.backends.candidate_logprobs import CandidateLogProbs
from spindle.backends.contract import ForwardBatch, ForwardItem, LossFn
from spindle.backends.megatron_runtime.common.chunked_logprobs import chunked_logprobs
from spindle.backends.megatron_runtime.common.forward_backward import (
    _loss,
    _vocab_parallel_logprobs,
    build_outputs,
    build_sequence_batches,
    pack_microbatches,
    pad_microbatches,
)


@pytest.mark.parametrize("cp_size", [1, 2])
def test_mixed_candidates_preserve_scores_gradients_and_shapes(cp_size: int) -> None:
    torch.manual_seed(31)
    datums = []
    references = []
    weight = torch.randn(13, 4, requires_grad=True)
    for length, width in ((11, None), (9, 1), (5, 3), (3, 5)):
        shape = (length,) if width is None else (length, width)
        targets = torch.zeros(shape, dtype=torch.long)
        targets[-1] = torch.randint(1, 13, shape[1:] or ())
        weights = torch.zeros(shape)
        weights[-1] = torch.randn(shape[1:] or ())
        if width == 3:
            targets[-1, 1] = targets[-1, 0]
            targets[-1, -1] = -100
        datum = Datum(
            ModelInput.from_ints([1] * length),
            {
                "target_tokens": targets,
                "weights": weights,
            },
        )
        datums.append(datum)
        references.append((targets, weights))
    assert datums[2].loss_fn_inputs["target_tokens"].sparse_crow_indices is not None
    batch = ForwardBatch((ForwardItem("m", tuple(datums)),), "cross_entropy")
    sequences = build_sequence_batches(batch, None, max_slots=None, max_seq_length=20)
    (packed,) = pack_microbatches(
        sequences, max_tokens=64, pad_to_multiple=2 * cp_size, total_pad_to_multiple=16
    )
    hidden = torch.randn(packed["tokens"].numel(), 4, requires_grad=True)
    output_collector = {0: []}
    total_loss = torch.tensor(0.0)
    tokens = 0
    for cp_rank in range(cp_size):
        index = torch.arange(hidden.shape[0])[cp_rank::cp_size]
        logprobs = chunked_logprobs(
            hidden[index], weight, packed["labels"][index], chunk=3, fp32=True
        )
        loss, count = _loss(
            logprobs, {**packed, "loss_mask": packed["loss_mask"][index]}
        )
        total_loss = total_loss + loss
        tokens += int(count)
        for output_index, start, length in zip(
            packed["output_indices"],
            packed["starts"],
            packed["original_lengths"],
            strict=True,
        ):
            selected = (index >= start) & (index < start + length)
            output_collector[0].append(
                {
                    "output_index": output_index,
                    "positions": (index[selected] - start).tolist(),
                    "logprobs": logprobs[selected].detach().tolist(),
                }
            )
    total_loss.backward()
    actual_hidden_grad, actual_weight_grad = hidden.grad.clone(), weight.grad.clone()
    hidden.grad = weight.grad = None
    (output,) = build_outputs(
        batch,
        output_collector,
        {
            0: {
                "loss": total_loss.item(),
                "tokens": float(tokens),
                "sequences": 4.0,
            }
        },
    )
    reference_loss = torch.tensor(0.0)
    expected_tokens = 0
    for start, (targets, weights), result in zip(
        packed["starts"], references, output.loss_fn_outputs, strict=True
    ):
        expected = (hidden[start : start + targets.shape[0]] @ weight.t()).log_softmax(
            -1
        )
        expected = expected.gather(
            1, targets.clamp_min(0).reshape(targets.shape[0], -1)
        )
        expected = expected.reshape(targets.shape).masked_fill(targets == -100, 0)
        torch.testing.assert_close(result["logprobs"].to_torch(), expected)
        reference_loss = reference_loss - (expected * weights).sum()
        expected_tokens += int(
            (weights != 0).reshape(targets.shape[0], -1).any(-1).sum()
        )
    reference_loss.backward()
    torch.testing.assert_close(total_loss, reference_loss)
    torch.testing.assert_close(actual_hidden_grad, hidden.grad)
    torch.testing.assert_close(actual_weight_grad, weight.grad)
    assert output.metrics["tokens:sum"] == expected_tokens == 4
    assert output.metrics["n_sequences:sum"] == 4
    (dummy,) = pad_microbatches([], 1, source=packed, dummy_tokens=4)
    dummy_scores = chunked_logprobs(
        torch.randn(4, 4), weight, dummy["labels"], chunk=3, fp32=True
    )
    dummy_loss, dummy_tokens = _loss(dummy_scores, dummy)
    assert dummy_loss.item() == dummy_tokens.item() == 0


@pytest.mark.parametrize("shape", [[3, 0], [2, 2], [3, 1, 1]])
def test_invalid_candidates_fail_before_packing(shape: list[int]) -> None:
    targets = torch.zeros(shape, dtype=torch.long)
    datum = Datum(
        ModelInput.from_ints([1, 2, 3]),
        {
            "target_tokens": TensorData.from_torch(targets),
        },
    )
    batch = ForwardBatch((ForwardItem("m", (datum,)),), "cross_entropy")
    with pytest.raises(ValueError, match="target_tokens"):
        build_sequence_batches(batch, None, max_slots=None, max_seq_length=8)


@pytest.mark.parametrize("loss_name", ["cross_entropy", "importance_sampling"])
def test_candidate_inputs_require_matching_weights_and_supported_loss(
    loss_name: LossFn,
) -> None:
    datum = Datum(
        ModelInput.from_ints([1, 2, 3]),
        {
            "target_tokens": torch.ones(3, 2, dtype=torch.long),
            "weights": [1.0, 0.0, 0.0],
        },
    )
    batch = ForwardBatch((ForwardItem("m", (datum,)),), loss_name)
    with pytest.raises(ValueError, match=r"weights.*shape|requires cross_entropy"):
        build_sequence_batches(batch, None, max_slots=None, max_seq_length=8)


def _check_vocab_partition(rank: int, rendezvous: str) -> None:
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2)
    try:
        torch.manual_seed(49)
        full = (torch.randn(5, 12) + 1000).requires_grad_()
        targets = torch.tensor(
            [[1, 8, 1], [9, 10, 11], [0, 0, 0], [2, 6, 8], [11, 3, 5]]
        )
        grad = torch.randn(5, 3)
        expected = full.log_softmax(-1).gather(1, targets)
        (expected * grad).sum().backward()
        shard = full.detach()[:, rank * 6 : (rank + 1) * 6].clone().requires_grad_()
        result = CandidateLogProbs.apply(shard, targets, dist.group.WORLD)
        (result * grad).sum().backward()
        torch.testing.assert_close(result, expected)
        torch.testing.assert_close(shard.grad, full.grad[:, rank * 6 : (rank + 1) * 6])
    finally:
        dist.destroy_process_group()


def test_candidate_tensor_parallel_gradients_match_full_vocabulary(
    tmp_path: Path,
) -> None:
    mp.spawn(_check_vocab_partition, args=(f"file://{tmp_path}/rendezvous",), nprocs=2)


def test_vocab_parallel_candidates_mask_ignored_labels(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    core = ModuleType("megatron.core")
    core.mpu = SimpleNamespace(get_tensor_model_parallel_group=lambda: None)
    monkeypatch.setitem(sys.modules, "megatron.core", core)
    logits = torch.randn(3, 9, requires_grad=True)
    reference = logits.detach().clone().requires_grad_()
    targets = torch.tensor([[2, 5, 2], [-100, 7, 8], [-100, -100, -100]])
    grad = torch.randn(3, 3)
    result = _vocab_parallel_logprobs(logits, targets)
    expected = (
        reference.log_softmax(-1)
        .gather(1, targets.clamp_min(0))
        .masked_fill(targets == -100, 0)
    )
    (result * grad).sum().backward()
    (expected * grad).sum().backward()
    torch.testing.assert_close(result, expected)
    torch.testing.assert_close(logits.grad, reference.grad)
