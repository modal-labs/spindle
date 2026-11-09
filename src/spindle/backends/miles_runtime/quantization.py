import torch


def pad_expert_slot_ids(slot_ids, tokens_per_expert, alignment, *, padded_size=None):
    padded_counts = (
        torch.div(tokens_per_expert + alignment - 1, alignment, rounding_mode="floor")
        * alignment
    )
    padding = padded_counts - tokens_per_expert
    shifts = torch.repeat_interleave(
        padding.cumsum(0) - padding, tokens_per_expert, output_size=slot_ids.numel()
    )
    destinations = torch.arange(slot_ids.numel(), device=slot_ids.device) + shifts
    if padded_size is None:
        padded_size = int(padded_counts.sum())
    padded_slots = slot_ids.new_zeros(padded_size)
    padded_slots.scatter_(0, destinations, slot_ids)
    return padded_slots, padded_counts


def validate_mxfp8_lora_model(chunks):
    # Megatron's native quantized tensor types exist only in the GPU image.
    from megatron.core.fp8_utils import is_grouped_mxfp8tensor, is_mxfp8tensor

    frozen_quantized = 0
    trainable = 0
    for chunk in chunks:
        for parameter in chunk.parameters():
            quantized = is_mxfp8tensor(parameter) or is_grouped_mxfp8tensor(parameter)
            if parameter.requires_grad:
                if quantized or parameter.dtype != torch.bfloat16:
                    raise RuntimeError(
                        "MXFP8 base LoRA requires BF16 adapter parameters"
                    )
                trainable += parameter.numel()
            elif quantized:
                frozen_quantized += parameter.numel()
    if not frozen_quantized or not trainable:
        raise RuntimeError(
            "MXFP8 base LoRA requires quantized frozen weights and trainable adapters"
        )
    return {
        "frozen_mxfp8_elements": frozen_quantized,
        "trainable_bf16_elements": trainable,
    }
