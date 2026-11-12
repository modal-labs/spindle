"""Miles v0.2.0 with the remaining Spindle trainer fixes."""

import hashlib

import modal

from .image_dependencies import (
    CORE_PACKAGES,
    MEGATRON_RUNTIME_CHECK,
    MEGATRON_RUNTIME_PACKAGES,
    STITCH_PACKAGE,
    ignore_config_source,
)

MILES_COMMIT = "8e419cc739fb09b5f21085aeacc7f6fd951084e1"
BASE_IMAGE = "radixark/miles:v0.2.0@sha256:c5e9ebe1dbb02faa094cf5036b57c4f3aadac9a53992f4d143823a6c20fb377c"
PATCH = '''\
diff --git a/miles/backends/megatron_utils/lora/bridge.py b/miles/backends/megatron_utils/lora/bridge.py
index 5b6ee82be..5baca2f31 100644
--- a/miles/backends/megatron_utils/lora/bridge.py
+++ b/miles/backends/megatron_utils/lora/bridge.py
@@ -121,7 +121,9 @@ def _validate_multi_lora_moe_support(args: TrainerConfig, provider) -> None:
     served = {target.rsplit(".", 1)[-1] for target in args.hf_lora_targets}
     if "gate_up_proj" in served:
         served.update(("gate_proj", "up_proj"))
-    expert_pair = {"gate_proj", "up_proj", "down_proj"}
+    expert_pair = {"up_proj", "down_proj"}
+    if provider.gated_linear_unit:
+        expert_pair.add("gate_proj")
     if served & expert_pair:
         assert expert_pair <= served, (
             f"Multi-LoRA on MoE experts requires all of {sorted(expert_pair)} in "
@@ -169,6 +171,7 @@ def _setup_lora_model_via_bridge(args: TrainerConfig) -> list:
     provider.sequence_parallel = args.backend.sequence_parallel
     provider.virtual_pipeline_model_parallel_size = args.backend.virtual_pipeline_model_parallel_size
     provider.context_parallel_size = args.backend.context_parallel_size
+    provider.cp_comm_type = args.backend.cp_comm_type[0] if len(args.backend.cp_comm_type) == 1 else args.backend.cp_comm_type
     provider.gradient_accumulation_fusion = args.backend.gradient_accumulation_fusion
     provider.recompute_granularity = args.backend.recompute_granularity
     provider.recompute_method = args.backend.recompute_method
diff --git a/miles/backends/megatron_utils/model_provider.py b/miles/backends/megatron_utils/model_provider.py
index bab1a02bd..82fe8891a 100644
--- a/miles/backends/megatron_utils/model_provider.py
+++ b/miles/backends/megatron_utils/model_provider.py
@@ -55,6 +55,7 @@ def _apply_bridge_runtime_config(provider, args: TrainerConfig) -> None:
     provider.expert_tensor_parallel_size = args.backend.expert_tensor_parallel_size
     provider.sequence_parallel = args.backend.sequence_parallel
     provider.context_parallel_size = args.backend.context_parallel_size
+    provider.cp_comm_type = args.backend.cp_comm_type[0] if len(args.backend.cp_comm_type) == 1 else args.backend.cp_comm_type
 
     # loss / sequence handling
     provider.calculate_per_token_loss = args.backend.calculate_per_token_loss  # CP>1 VL models assert this
diff --git a/miles/backends/megatron_utils/update_weight/hf_weight_iterator_bridge.py b/miles/backends/megatron_utils/update_weight/hf_weight_iterator_bridge.py
index f42c266fc..66fcdcba8 100644
--- a/miles/backends/megatron_utils/update_weight/hf_weight_iterator_bridge.py
+++ b/miles/backends/megatron_utils/update_weight/hf_weight_iterator_bridge.py
@@ -103,7 +103,7 @@ class HfWeightIteratorBridge(MegatronHfWeightIteratorBase):
         return [(h, slice_lora_to_rank(h, w, adapter.rank)) for h, w in named_tensors]
 
     def _export_current_adapter(self) -> list:
-        return list(self.iter_current_adapter())
+        return [(name, tensor.detach().to("cpu", copy=True)) for name, tensor in self.iter_current_adapter()]
 
     def iter_current_adapter(self):
         """The exposed adapter's HF-named tensors, TP/EP gathered; Bridge broadcasts every PP stage's adapters to all stages."""
'''
PATCH_SHA256 = hashlib.sha256(PATCH.encode()).hexdigest()

image = (
    modal.Image.from_registry(BASE_IMAGE)
    .entrypoint([])
    .env(
        {
            "LD_LIBRARY_PATH": "/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH",
            "PYTHONPATH": "/root/Megatron-LM:$PYTHONPATH",
            "CUDA_DEVICE_MAX_CONNECTIONS": "1",
            "SPINDLE_MILES_API": "0.2",
            "SPINDLE_HOST_SYNC_PATCHES": "0",
            "SPINDLE_MILES_PATCH_SHA256": PATCH_SHA256,
        }
    )
    .pip_install(*CORE_PACKAGES, STITCH_PACKAGE, *MEGATRON_RUNTIME_PACKAGES)
    .run_commands(
        f'test "$(git -C /root/miles rev-parse HEAD)" = "{MILES_COMMIT}"',
        "git -C /root/miles apply - <<'SPINDLE_PATCH'\n" + PATCH + "SPINDLE_PATCH",
        MEGATRON_RUNTIME_CHECK,
    )
    .add_local_python_source("spindle", ignore=ignore_config_source, copy=True)
    .run_commands(
        "python -c 'from spindle.backends.miles_runtime import v020, actor_v020'"
    )
)
