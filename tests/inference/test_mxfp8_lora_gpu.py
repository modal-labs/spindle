from types import SimpleNamespace

import pytest
import torch


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10,
    reason="requires Blackwell and DeepGEMM",
)
@pytest.mark.parametrize("masked", [False, True])
@pytest.mark.parametrize("precision", ["bf16", "mxfp8"])
def test_native_moe_applies_bf16_lora_before_activation_and_router_weights(
    monkeypatch, masked, precision
):
    # SGLang, its kernels, and DeepGEMM are installed only in the GPU image.
    import deep_gemm.utils.layout
    from sglang.srt.layers.quantization.fp8_utils import mxfp8_group_quantize
    from sglang.srt.layers.moe.moe_runner import deep_gemm as dg
    from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
    from sglang.srt.layers.moe.moe_runner.runner import MoeRunner
    from sglang.srt.layers.moe.topk import StandardTopKOutput
    from sglang.srt.layers.moe.token_dispatcher.standard import StandardDispatchOutput
    from sglang.srt.layers.moe.utils import MoeRunnerBackend
    from sglang.srt.lora.lora_moe_runners import LoRAInfo
    from sglang.srt.runtime_context import get_flags, get_parallel

    torch.manual_seed(42)
    device = "cuda"
    tokens, experts, hidden, intermediate, rank = 8, 4, 128, 128, 16
    w13 = (
        torch.randn(
            experts, 2 * intermediate, hidden, device=device, dtype=torch.bfloat16
        )
        * 0.02
    )
    w2 = (
        torch.randn(experts, hidden, intermediate, device=device, dtype=torch.bfloat16)
        * 0.02
    )

    def quantize(weight):
        q, scales = mxfp8_group_quantize(
            weight.reshape(-1, weight.shape[-1]).contiguous()
        )
        scales = torch.pow(2.0, scales.float() - 127).reshape(
            experts, weight.shape[1], -1
        )
        dequantized = (
            q.reshape(*weight.shape).float() * scales.repeat_interleave(32, -1)
        ).to(torch.bfloat16)
        packed = deep_gemm.utils.layout.get_mn_major_tma_aligned_packed_ue8m0_tensor(
            scales
        )
        return q.reshape_as(weight), packed, dequantized

    if precision == "mxfp8":
        q13, s13, w13 = quantize(w13)
        q2, s2, w2 = quantize(w2)
        quant = dg.DeepGemmMoeQuantInfo(
            q13,
            q2,
            True,
            w13_scale=s13,
            w2_scale=s2,
            block_shape=[1, 32],
            use_mxfp8=True,
        )
    else:
        quant = dg.DeepGemmMoeQuantInfo(w13, w2, False)
    x = torch.randn(tokens, hidden, device=device, dtype=torch.bfloat16)
    ids = torch.stack(
        [
            torch.arange(tokens, device=device) % experts,
            (torch.arange(tokens, device=device) + 1) % experts,
        ],
        dim=-1,
    ).int()
    router_weights = (
        torch.tensor([0.2, 0.8], device=device).expand(tokens, 2).contiguous()
    )
    slot = torch.arange(tokens, device=device) // 4
    info = LoRAInfo(
        gate_up_lora_a_weights=torch.randn(
            2, experts, 2 * rank, hidden, device=device, dtype=torch.bfloat16
        )
        * 0.05,
        gate_up_lora_b_weights=torch.randn(
            2, experts, 2 * intermediate, rank, device=device, dtype=torch.bfloat16
        )
        * 0.05,
        down_lora_a_weights=torch.randn(
            2, experts, rank, intermediate, device=device, dtype=torch.bfloat16
        )
        * 0.05,
        down_lora_b_weights=torch.randn(
            2, experts, hidden, rank, device=device, dtype=torch.bfloat16
        )
        * 0.05,
        seg_indptr=torch.tensor([0, 4, 8], device=device, dtype=torch.int32),
        req_to_lora=torch.tensor([0, 1], device=device, dtype=torch.int32),
        lora_ranks=torch.tensor([rank, rank], device=device, dtype=torch.int32),
        adapter_enabled=torch.ones(2, device=device, dtype=torch.int32),
        token_lora_mapping=slot.int(),
        max_lora_rank=rank,
        num_experts=experts,
    )
    config = MoeRunnerConfig(
        num_experts=experts,
        num_local_experts=experts,
        hidden_size=hidden,
        intermediate_size_per_partition=intermediate,
        top_k=2,
        inplace=False,
        gate_up_interleaved=False,
    )
    monkeypatch.setattr(dg, "_should_use_masked_standard_layout", lambda *a: masked)
    monkeypatch.setattr(dg, "is_allocation_symmetric", lambda: False)
    monkeypatch.setattr(
        dg,
        "get_exec",
        lambda: SimpleNamespace(
            deterministic=SimpleNamespace(enable_deterministic_inference=False)
        ),
    )
    monkeypatch.setattr(get_flags().moe, "runner_backend", MoeRunnerBackend.DEEP_GEMM)
    with get_parallel().override(tp_size=1, tp_group=None, ep_size=1):
        runner = MoeRunner(MoeRunnerBackend.DEEP_GEMM, config, lora_enabled=True)
        dispatch = StandardDispatchOutput(
            x, None, StandardTopKOutput(router_weights, ids, None)
        )
        actual = runner.run(dispatch, quant, lora_info=info).hidden_states
    expected = torch.zeros_like(x)
    for token in range(tokens):
        for choice in range(2):
            expert = int(ids[token, choice])
            adapter = int(slot[token])
            up = x[token] @ w13[expert].T
            up[:intermediate] += (
                x[token] @ info.gate_up_lora_a_weights[adapter, expert, :rank].T
            ) @ info.gate_up_lora_b_weights[adapter, expert, :intermediate].T
            up[intermediate:] += (
                x[token] @ info.gate_up_lora_a_weights[adapter, expert, rank:].T
            ) @ info.gate_up_lora_b_weights[adapter, expert, intermediate:].T
            middle = (
                torch.nn.functional.silu(up[:intermediate].float())
                * up[intermediate:].float()
            ).to(torch.bfloat16)
            value = middle @ w2[expert].T
            value += (
                middle @ info.down_lora_a_weights[adapter, expert].T
            ) @ info.down_lora_b_weights[adapter, expert].T
            expected[token] += value * router_weights[token, choice]
    torch.testing.assert_close(
        actual,
        expected,
        atol=0.004 if precision == "mxfp8" else 0.001,
        rtol=0.15 if precision == "mxfp8" else 0.04,
    )
