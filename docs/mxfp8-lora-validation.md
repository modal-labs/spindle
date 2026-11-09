# MXFP8 frozen base with BF16 LoRA

`qwen35_35b_a3b_mxfp8_lora_8k` uses MXFP8 frozen base projections in the
Miles LoRA trainer and SGLang sampler. Trainable adapter parameters and adapter
publications stay BF16. `qwen35_35b_a3b_lora_8k` is the matched BF16 control.
Both use Qwen3.5-35B-A3B, rank 16, four B300 trainer GPUs, one B300 sampler GPU,
and an 8,192-token context. The model revision is pinned in both recipes.

Set `miles_cfg.base_quantization="mxfp8"` for the frozen trainer base and
`sglang_cfg.quantization="mxfp8"` for the sampler. The trainer verifies native
MXFP8 storage and BF16 trainable parameters before accepting work. Embeddings, the vision tower,
and normalization parameters retain their normal precision. The sampler also
keeps the 64-channel `in_proj_ba` projection in BF16.

Use the stock Tinker SDK's `create_lora_training_client` with rank 16 and
`train_unembed=False`. The recipes target attention and routed/shared MLP
projections, including expert adapters. They do not train the output head.

The trainer patch maps adapter slots onto MXFP8's padded expert rows. The
serving patch applies BF16 adapter deltas after the gate/up projection, before
SiLU, and after the down projection, before routing weights are applied.
DeepGEMM retains native MXFP8 base GEMMs. Both masked decode and packed prefill
layouts are covered. The current serving path requires expert parallel size 1
and unclamped, unswizzled SiLU intermediates.

CPU tests compare padded adapter outputs and gradients with an unpadded
reference, including empty experts. Serving layout tests cover invalid routing
entries and check that routing weights are applied once. A B300 native expert
test passed with frozen MXFP8 weights, BF16 adapter gradients matching the
reference, and finite input gradients.

Full-model LoRA training, adapter publication, checkpoint restart, and the
4,096-token decode comparison are being validated. No performance or learning
quality result is established by the layer tests.
