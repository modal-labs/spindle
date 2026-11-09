BRIDGE_PATCH = (
    "diff --git a/src/megatron/bridge/peft/multi_lora_layers.py b/src/megatron/bridge/peft/multi_lora_layers.py\n"
    "index 2e7f797b..1b6748fd 100644\n"
    "--- a/src/megatron/bridge/peft/multi_lora_layers.py\n"
    "+++ b/src/megatron/bridge/peft/multi_lora_layers.py\n"
    "@@ -49,6 +49,7 @@ from megatron.core.tensor_parallel.mappings import (\n"
    " from megatron.core.transformer.moe.moe_utils import sort_chunks_by_idxs\n"
    " \n"
    " from megatron.bridge.peft.adapter_wrapper import AdapterWrapper\n"
    "+from spindle.backends.miles_runtime.quantization import pad_expert_slot_ids\n"
    " from megatron.bridge.peft.utils import (\n"
    "     GroupedExpertLinearAdapter,\n"
    "     ParallelLinearAdapter,\n"
    "@@ -515,13 +516,14 @@ class MultiLoRAGroupedExpertLinear(MultiLoRALinear):\n"
    '                 f"Multi-LoRA on MoE experts requires expert_tensor_parallel_size=1 "\n'
    '                 f"(got {expert_tp_size}) for {full_name}."\n'
    "             )\n"
    "-        # TEGroupedMLP pads its input to the quantization alignment before\n"
    "-        # calling the grouped linears, which would desynchronize the row order\n"
    "-        # the slot routing was built for.\n"
    '-        if getattr(config, "fp8", None) or getattr(config, "fp4", None):\n'
    "+        # MXFP8 slot routing includes the expert module's 32-row padding.\n"
    '+        fp8_recipe = getattr(config, "fp8_recipe", None)\n'
    '+        mxfp8 = getattr(fp8_recipe, "value", fp8_recipe) == "mxfp8"\n'
    '+        if mxfp8 and getattr(config, "use_transformer_engine_op_fuser", False):\n'
    '+            raise NotImplementedError("MXFP8 expert LoRA requires the unfused expert module.")\n'
    '+        if getattr(config, "fp4", None) or (getattr(config, "fp8", None) and not mxfp8):\n'
    "             raise NotImplementedError(\n"
    '-                f"Multi-LoRA on MoE experts does not support fp8/fp4 expert quantization "\n'
    '-                f"(quantization padding changes the dispatched token order) for {full_name}."\n'
    '+                f"Multi-LoRA on quantized MoE experts requires the MXFP8 recipe for {full_name}."\n'
    "             )\n"
    "         # The remaining dispatch requirements are checked here, at model build\n"
    "         # time, rather than in the routing hook: a hook that raises on only some\n"
    "@@ -829,6 +831,7 @@ def _build_expert_slot_routing(\n"
    "     n_adapters: int,\n"
    "     num_local_experts: int,\n"
    "     device: torch.device,\n"
    "+    quantization_alignment: int = 1,\n"
    " ) -> ExpertSlotRouting:\n"
    '     """Derive the per-(slot, expert) segmentation of one MoE layer\'s dispatched tokens.\n'
    " \n"
    "@@ -874,6 +877,10 @@ def _build_expert_slot_routing(\n"
    '             f"but the co-permuted slot ids cover {slot_ids.shape[0]} tokens."\n'
    "         )\n"
    " \n"
    "+    if quantization_alignment > 1:\n"
    "+        slot_ids, padded_counts = pad_expert_slot_ids(slot_ids, per_expert, quantization_alignment)\n"
    "+        expert_ids = torch.repeat_interleave(torch.arange(num_local_experts, device=device), padded_counts)\n"
    "+\n"
    "     # Slot-major key so a stable sort yields contiguous (slot, expert) groups\n"
    "     # while preserving the dispatcher's expert order inside each slot.\n"
    "     keys = slot_ids.long() * num_local_experts + expert_ids\n"
    "@@ -920,6 +927,9 @@ def _make_slot_routing_hook(moe_layer: nn.Module, expert_layers: List[MultiLoRAG\n"
    '                 "set_tokens_per_adapter_slot(model, adapter_token_counts) must run before every "\n'
    '                 "forward when MoE experts carry multi-LoRA adapters."\n'
    "             )\n"
    "+        config = reference.to_wrap.config\n"
    '+        recipe = getattr(config, "fp8_recipe", None)\n'
    '+        alignment = 32 if config.fp8 and getattr(recipe, "value", recipe) == "mxfp8" else 1\n'
    "         routing = _build_expert_slot_routing(\n"
    "             moe_layer.token_dispatcher,\n"
    "             tokens_per_expert,\n"
    "@@ -927,6 +937,7 @@ def _make_slot_routing_hook(moe_layer: nn.Module, expert_layers: List[MultiLoRAG\n"
    "             reference.n_adapters,\n"
    "             reference.num_local_experts,\n"
    "             hidden_states.device,\n"
    "+            quantization_alignment=alignment,\n"
    "         )\n"
    "         for layer in expert_layers:\n"
    "             layer.expert_slot_routing = routing\n"
)

MILES_PATCH = (
    "diff --git a/miles/backends/megatron_utils/lora/bridge.py b/miles/backends/megatron_utils/lora/bridge.py\n"
    "index 4cd3faa7..ae8a5820 100644\n"
    "--- a/miles/backends/megatron_utils/lora/bridge.py\n"
    "+++ b/miles/backends/megatron_utils/lora/bridge.py\n"
    "@@ -8,9 +8,13 @@ from __future__ import annotations\n"
    " \n"
    " import logging\n"
    " from argparse import Namespace\n"
    "+from contextlib import nullcontext\n"
    " from dataclasses import dataclass\n"
    " \n"
    " import torch.distributed as dist\n"
    "+from transformer_engine.common.recipe import MXFP8BlockScaling\n"
    "+from transformer_engine.pytorch import quantized_model_init\n"
    "+from spindle.backends.miles_runtime.quantization import validate_mxfp8_lora_model\n"
    " from megatron.core.tensor_parallel import ColumnParallelLinear\n"
    " from megatron.core.transformer.moe.router import TopKRouter\n"
    " from megatron.core.utils import get_attr_wrapped_model\n"
    "@@ -26,6 +30,8 @@ from miles.utils.hf_utils.weight_mapping import HfWeightMapping\n"
    " from miles.utils.lora.utils import is_multi_lora_enabled, targets_expert_leaves\n"
    " from miles.utils.megatron_bridge_utils import apply_dsa_backend_args\n"
    " \n"
    "+SPINDLE_MXFP8_BASE_SUPPORTED = True\n"
    "+\n"
    " logger = logging.getLogger(__name__)\n"
    " \n"
    " \n"
    "@@ -97,9 +103,9 @@ def _validate_multi_lora_moe_support(args: Namespace, provider) -> None:\n"
    '         "Multi-LoRA on MoE experts requires moe_grouped_gemm=True (SequentialMLP expert "\n'
    '         "linears are skipped, so the experts would train no adapter)."\n'
    "     )\n"
    '-    assert not getattr(provider, "fp8", None) and not getattr(provider, "fp4", None), (\n'
    '-        "Multi-LoRA on MoE experts does not support fp8/fp4 experts (quantization padding "\n'
    '-        "desynchronizes the dispatched token order)."\n'
    '+    recipe = getattr(provider.fp8_recipe, "value", provider.fp8_recipe)\n'
    '+    assert not getattr(provider, "fp4", None) and (not provider.fp8 or recipe == "mxfp8"), (\n'
    '+        "Multi-LoRA on quantized MoE experts requires the MXFP8 recipe."\n'
    "     )\n"
    "     # sglang only wraps a fused MoE layer when both expert projections are targeted.\n"
    '     served = {target.rsplit(".", 1)[-1] for target in args.hf_lora_targets}\n'
    "@@ -146,6 +152,11 @@ def _setup_lora_model_via_bridge(args: Namespace) -> list:\n"
    "     bridge = AutoBridge.from_hf_pretrained(args.hf_checkpoint, trust_remote_code=True)\n"
    "     provider = bridge.to_megatron_provider(load_weights=False)\n"
    " \n"
    '+    if getattr(args, "spindle_base_quantization", None) == "mxfp8":\n'
    '+        provider.fp8 = "hybrid"\n'
    '+        provider.fp8_recipe = "mxfp8"\n'
    "+        provider.fp8_param = True\n"
    "+        provider.use_transformer_engine_op_fuser = False\n"
    "     provider.tensor_model_parallel_size = args.tensor_model_parallel_size\n"
    "     provider.pipeline_model_parallel_size = args.pipeline_model_parallel_size\n"
    "     provider.expert_model_parallel_size = args.expert_model_parallel_size\n"
    "@@ -255,5 +266,13 @@ def _setup_lora_model_via_bridge(args: Namespace) -> list:\n"
    "     if args.offload_train:\n"
    "         patch_param_grad_buffer_for_colocate_mode_lora()\n"
    " \n"
    "-    model = provider.provide_distributed_model(wrap_with_ddp=True, ddp_config=ddp_config)\n"
    "+    model_context = (\n"
    "+        quantized_model_init(enabled=True, recipe=MXFP8BlockScaling(), preserve_high_precision_init_val=False)\n"
    '+        if getattr(args, "spindle_base_quantization", None) == "mxfp8"\n'
    "+        else nullcontext()\n"
    "+    )\n"
    "+    with model_context:\n"
    "+        model = provider.provide_distributed_model(wrap_with_ddp=True, ddp_config=ddp_config)\n"
    '+    if getattr(args, "spindle_base_quantization", None) == "mxfp8":\n'
    '+        logger.info("MXFP8 base LoRA storage: %s", validate_mxfp8_lora_model(model))\n'
    "     return model\n"
)
