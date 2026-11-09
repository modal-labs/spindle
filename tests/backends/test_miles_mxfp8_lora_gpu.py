import os

import pytest
import torch

from spindle.backends.miles_runtime.quantization import pad_expert_slot_ids


@pytest.fixture
def model_parallel(tmp_path):
    # Megatron process groups exist only in the GPU test image.
    from megatron.core import parallel_state
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed

    os.environ.setdefault("CUDA_DEVICE_MAX_CONNECTIONS", "1")
    torch.distributed.init_process_group(
        "nccl", init_method=f"file://{tmp_path / 'nccl'}", rank=0, world_size=1
    )
    parallel_state.initialize_model_parallel()
    model_parallel_cuda_manual_seed(42)
    try:
        yield
    finally:
        parallel_state.destroy_model_parallel()
        torch.distributed.destroy_process_group()


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10,
    reason="requires Blackwell and Transformer Engine",
)
def test_mxfp8_frozen_experts_keep_bf16_lora_gradients(model_parallel):
    # Megatron and Transformer Engine are installed only in the GPU image.
    from transformer_engine.common.recipe import MXFP8BlockScaling
    from transformer_engine.pytorch import quantized_model_init
    from megatron.core.extensions.transformer_engine import (
        TEColumnParallelGroupedLinear,
    )
    from megatron.core.fp8_utils import (
        get_fp8_context,
        is_grouped_mxfp8tensor,
        is_mxfp8tensor,
    )
    from megatron.core.process_groups_config import ProcessGroupCollection
    from megatron.core.transformer.transformer_config import TransformerConfig
    from megatron.bridge.peft.multi_lora_layers import (
        ExpertSlotRouting,
        MultiLoRAGroupedExpertLinear,
    )

    config = TransformerConfig(
        num_layers=1,
        hidden_size=128,
        num_attention_heads=4,
        num_moe_experts=4,
        moe_ffn_hidden_size=128,
        moe_grouped_gemm=True,
        moe_token_dispatcher_type="alltoall",
        moe_permute_fusion=False,
        fp8="hybrid",
        fp8_recipe="mxfp8",
        fp8_param=True,
        bf16=True,
        params_dtype=torch.bfloat16,
        gradient_accumulation_fusion=False,
        add_bias_linear=False,
    )
    with quantized_model_init(enabled=True, recipe=MXFP8BlockScaling()):
        base = TEColumnParallelGroupedLinear(
            4,
            128,
            128,
            config=config,
            init_method=config.init_method,
            bias=False,
            skip_bias_add=False,
            is_expert=True,
            pg_collection=ProcessGroupCollection.use_mpu_process_groups(),
            name="decoder.layers.0.mlp.experts.linear_fc1",
        )
    base.finish_init(None)
    base.requires_grad_(False)
    quantized = [
        p for p in base.parameters() if is_mxfp8tensor(p) or is_grouped_mxfp8tensor(p)
    ]
    assert quantized
    layer = MultiLoRAGroupedExpertLinear(
        base,
        n_adapters=2,
        dim=16,
        alpha=16,
        full_name="decoder.layers.0.mlp.experts.linear_fc1",
        num_local_experts=4,
    )
    layer.alpha_values.fill_(16)
    for adapter in layer.adapters:
        torch.nn.init.normal_(adapter.linear_out.weight, std=0.05)
    counts = torch.tensor([3, 0, 17, 8], device="cuda")
    slots = torch.arange(28, device="cuda", dtype=torch.int32) % 2
    padded_slots, padded_counts = pad_expert_slot_ids(slots, counts, 32)
    x = torch.randn(28, 128, device="cuda", dtype=torch.bfloat16)
    padded_x = torch.cat(
        [
            torch.nn.functional.pad(chunk, (0, 0, 0, int(padded - count)))
            for chunk, count, padded in zip(
                x.split(counts.tolist()), counts, padded_counts, strict=True
            )
        ]
    )
    padded_x.requires_grad_()
    expert_ids = torch.repeat_interleave(torch.arange(4, device="cuda"), padded_counts)
    keys = padded_slots.long() * 4 + expert_ids
    order = keys.argsort(stable=True)
    inverse = order.argsort()
    group_counts = torch.bincount(keys, minlength=8)
    layer.expert_slot_routing = ExpertSlotRouting(
        sort_idx=order,
        inverse_idx=inverse,
        group_offsets=group_counts.cumsum(0, dtype=torch.int32),
        slot_token_counts=group_counts.view(2, 4).sum(1),
        num_tokens=len(keys),
    )
    with get_fp8_context(config):
        baseline, _ = base(padded_x, padded_counts.tolist())
        actual, _ = layer(padded_x, padded_counts.tolist())
    a = (
        torch.stack([adapter.linear_in.weight for adapter in layer.adapters])
        .detach()
        .requires_grad_()
    )
    b = (
        torch.stack([adapter.linear_out.weight for adapter in layer.adapters])
        .detach()
        .requires_grad_()
    )
    reference_mid = torch.bmm(
        padded_x[:, None, :], a[padded_slots.long(), expert_ids].transpose(1, 2)
    )
    reference_delta = torch.bmm(
        reference_mid, b[padded_slots.long(), expert_ids].transpose(1, 2)
    ).squeeze(1)
    expected = baseline.detach() + reference_delta
    torch.testing.assert_close(actual, expected, atol=0.004, rtol=0.02)
    actual.float().square().sum().backward()
    expected.float().square().sum().backward()
    for slot, adapter in enumerate(layer.adapters):
        assert adapter.linear_in.weight.dtype == torch.bfloat16
        assert adapter.linear_out.weight.dtype == torch.bfloat16
        torch.testing.assert_close(
            adapter.linear_in.weight.grad, a.grad[slot], atol=0.03, rtol=0.04
        )
        torch.testing.assert_close(
            adapter.linear_out.weight.grad, b.grad[slot], atol=0.03, rtol=0.04
        )
    assert all(p.grad is None for p in base.parameters())
    assert padded_x.grad is not None and torch.isfinite(padded_x.grad).all()
    from spindle.backends.miles_runtime.quantization import validate_mxfp8_lora_model

    summary = validate_mxfp8_lora_model([layer])
    assert summary["frozen_mxfp8_elements"] > 0
    assert summary["trainable_bf16_elements"] > 0
    print("MXFP8 base parameters:", [type(p).__name__ for p in quantized])
    print(
        "Init-state methods:",
        [name for name in dir(quantized[0]) if "init_val" in name],
    )


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10,
    reason="requires Blackwell and Transformer Engine",
)
def test_vision_stays_bf16_without_disabling_text_mxfp8(model_parallel):
    # The real vision model and Transformer Engine are GPU-image dependencies.
    from transformer_engine.common.recipe import MXFP8BlockScaling
    from transformer_engine.pytorch import Linear, quantized_model_init
    from megatron.core.extensions.transformer_engine import (
        TEColumnParallelLinear,
        TENorm,
        TERowParallelLinear,
    )
    from megatron.core.fp8_utils import is_grouped_mxfp8tensor, is_mxfp8tensor
    from megatron.core.models.vision.vit_layer_specs import (
        get_vit_layer_with_transformer_engine_spec,
    )
    from megatron.core.process_groups_config import ProcessGroupCollection
    from megatron.bridge.models.qwen_vl.modelling_qwen3_vl.transformer_config import (
        Qwen3VLTransformerConfig,
    )
    from megatron.bridge.models.qwen_vl.modelling_qwen3_vl.utils import (
        PatchMergerSubmodules,
    )
    from megatron.bridge.models.qwen_vl.modelling_qwen3_vl.vision_model import (
        Qwen3VLVisionModel,
    )

    config = Qwen3VLTransformerConfig(
        num_layers=1,
        hidden_size=128,
        num_attention_heads=4,
        ffn_hidden_size=130,
        bf16=True,
        params_dtype=torch.bfloat16,
        normalization="LayerNorm",
        out_hidden_size=128,
        num_position_embeddings=16,
        deepstack_visual_indexes=[],
        gradient_accumulation_fusion=False,
    )
    spec = get_vit_layer_with_transformer_engine_spec()
    merger = PatchMergerSubmodules(
        patch_norm=TENorm,
        linear_fc1=TEColumnParallelLinear,
        linear_fc2=TERowParallelLinear,
    )
    with quantized_model_init(enabled=True, recipe=MXFP8BlockScaling()):
        vision = Qwen3VLVisionModel(
            config,
            spec,
            merger,
            pg_collection=ProcessGroupCollection.use_mpu_process_groups(),
        )
        language = Linear(
            128, 128, params_dtype=torch.bfloat16, device="cuda", bias=False
        )
    assert any(p.ndim == 2 and 130 in p.shape for p in vision.parameters())
    assert all(
        not is_mxfp8tensor(p) and not is_grouped_mxfp8tensor(p)
        for p in vision.parameters()
    )
    assert is_mxfp8tensor(language.weight)
