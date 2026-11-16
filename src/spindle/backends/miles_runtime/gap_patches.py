"""Prototype kernel-time patches, each opt-in through SPINDLE_GAP_PATCHES (comma list).

- relu2: fuse the MoE squared-ReLU with the router-probability multiply
  (Megatron's use_fused_weighted_squared_relu).
- logits_split: take every sample's response logits from one split of the
  micro-batch logits, so backward builds the logits gradient once instead of
  once per sample.
- permute_bwd: MoE token permute whose backward is a scatter-add, not a BF16
  index_add_.
- lora_grad: give stacked expert-adapter weights contiguous gradients.
- permute_fusion: let the MoE dispatcher use Transformer Engine's fused permute
  and unpermute while expert adapters are on; slot ids follow the same order
  through an index rebuilt from the routing map.
- expert_grouped_mm: run frozen MoE expert weights through one torch grouped GEMM
  per projection instead of Transformer Engine's per-expert cuBLAS calls.
"""

from __future__ import annotations

import os

import torch

ENV = "SPINDLE_GAP_PATCHES"


def enabled() -> set[str]:
    return {p.strip() for p in os.environ.get(ENV, "").split(",") if p.strip()}


def _log(**fields) -> None:
    print("spindle_gap_patches " + " ".join(f"{k}={v}" for k, v in fields.items()), flush=True)


def install(logit_processors) -> None:
    on = enabled()
    if "relu2" in on or "permute_fusion" in on:
        _install_provider_overrides(relu2="relu2" in on, permute_fusion="permute_fusion" in on)
    if "permute_fusion" in on:
        _install_permute_fusion()
    if "logits_split" in on:
        _install_logits_split(logit_processors)
    if "permute_bwd" in on:
        _install_permute_bwd()
    if "expert_grouped_mm" in on:
        _install_expert_grouped_mm()
    if on:
        _log(action="installed", patches=",".join(sorted(on)))


def lora_grad_enabled() -> bool:
    return "lora_grad" in enabled()


def _install_provider_overrides(*, relu2: bool, permute_fusion: bool) -> None:
    from megatron.bridge import AutoBridge

    original = AutoBridge.to_megatron_provider

    def to_megatron_provider(bridge, *args, **kwargs):
        provider = original(bridge, *args, **kwargs)
        if relu2:
            provider.use_fused_weighted_squared_relu = True
        if permute_fusion:
            # Miles turns the fusion off right before finalize; turn it back on there.
            finalize = provider.finalize

            def finalize_with_fusion(*a, **k):
                provider.moe_permute_fusion = True
                return finalize(*a, **k)

            provider.finalize = finalize_with_fusion
        return provider

    AutoBridge.to_megatron_provider = to_megatron_provider


def _unfused_permute_index(routing_map: torch.Tensor, num_out_tokens: int) -> torch.Tensor:
    num_tokens = routing_map.shape[0]
    flat = routing_map.bool().T.contiguous().reshape(-1).argsort(descending=True, stable=True)
    return flat[:num_out_tokens] % num_tokens


def _install_permute_fusion() -> None:
    from megatron.bridge.peft import multi_lora_layers as layers
    from megatron.core.transformer.moe import moe_utils, token_dispatcher

    init = layers.MultiLoRAGroupedExpertLinear.__init__

    def __init__(self, to_wrap, *args, **kwargs):
        config = to_wrap.config
        fused = config.moe_permute_fusion
        config.moe_permute_fusion = False
        try:
            init(self, to_wrap, *args, **kwargs)
        finally:
            config.moe_permute_fusion = fused

    layers.MultiLoRAGroupedExpertLinear.__init__ = __init__

    co_permute = layers._co_permute_slot_ids

    def _co_permute_slot_ids(dispatcher, slot_ids, num_local_experts):
        if not dispatcher.config.moe_permute_fusion:
            return co_permute(dispatcher, slot_ids, num_local_experts)
        if getattr(dispatcher, "ep_size", 1) > 1 or getattr(dispatcher, "tp_size", 1) > 1:
            raise NotImplementedError("permute_fusion prototype covers expert parallel size 1 only")
        index = _unfused_permute_index(dispatcher.routing_map, dispatcher.num_out_tokens)
        slot_ids = slot_ids.index_select(0, index)
        if num_local_experts > 1:
            slot_ids, _ = layers.sort_chunks_by_idxs(
                slot_ids,
                dispatcher.num_global_tokens_per_local_expert.ravel(),
                dispatcher.sort_input_by_local_experts,
                fused=False,
            )
        return slot_ids

    layers._co_permute_slot_ids = _co_permute_slot_ids

    permute = moe_utils.permute
    checked = []

    def checked_permute(tokens, routing_map, probs=None, num_out_tokens=None, fused=False, *args, **kwargs):
        out = permute(tokens, routing_map, probs, num_out_tokens, fused, *args, **kwargs)
        if fused and not checked and num_out_tokens is not None:
            checked.append(True)
            with torch.no_grad():
                index = _unfused_permute_index(routing_map, num_out_tokens)
                same = bool(torch.equal(out[0], tokens.index_select(0, index)))
            _log(action="fused_permute_order_check", equal=same)
        return out

    moe_utils.permute = checked_permute
    token_dispatcher.permute = checked_permute


def _install_logits_split(processors) -> None:
    original = processors._iter_response_chunks

    def _iter_response_chunks(logits, *, args, unconcat_tokens, total_lengths,
                              response_lengths, max_seq_lens=None, include_response_indices):
        windows = None
        if (
            args.qkv_format == "thd"
            and not args.true_on_policy_mode
            and processors.get_parallel_state().cp.size == 1
            and logits.dim() == 3
            and logits.size(0) == 1
        ):
            windows = _response_windows(total_lengths, response_lengths, logits.size(1))
        if windows is None:
            yield from original(
                logits, args=args, unconcat_tokens=unconcat_tokens, total_lengths=total_lengths,
                response_lengths=response_lengths, max_seq_lens=max_seq_lens,
                include_response_indices=include_response_indices,
            )
            return
        sizes, keep = windows
        pieces = logits.squeeze(0).split(sizes)
        for index, tokens, response_length in zip(keep, unconcat_tokens, response_lengths, strict=False):
            tokens_chunk = tokens[-response_length:] if response_length else tokens[0:0]
            indices = range(response_length) if include_response_indices else ()
            yield pieces[index], tokens_chunk, indices

    processors._iter_response_chunks = _iter_response_chunks


def _response_windows(total_lengths, response_lengths, rows):
    """Split sizes covering [0, rows) and the piece index of each sample's window."""
    sizes, keep, cursor, end = [], [], 0, 0
    for total, response in zip(total_lengths, response_lengths, strict=False):
        end += total
        lo, hi = end - response - 1, end - 1
        if response == 0:
            lo = hi = max(cursor, min(hi, rows))
        if lo < cursor or hi > rows or lo > hi:
            return None
        if lo > cursor:
            sizes.append(lo - cursor)
        keep.append(len(sizes))
        sizes.append(hi - lo)
        cursor = hi
    if cursor < rows:
        sizes.append(rows - cursor)
    return sizes, keep


class _PermuteRows(torch.autograd.Function):
    """``tokens.index_select(0, index)`` whose backward scatter-adds rows."""

    @staticmethod
    def forward(ctx, tokens, index):
        ctx.save_for_backward(index)
        ctx.rows = tokens.size(0)
        return tokens.index_select(0, index)

    @staticmethod
    def backward(ctx, grad):
        (index,) = ctx.saved_tensors
        out = grad.new_zeros((ctx.rows, grad.size(1)))
        out.scatter_add_(0, index.unsqueeze(1).expand(-1, grad.size(1)), grad.contiguous())
        return out, None


def _install_permute_bwd() -> None:
    from megatron.core.transformer.moe import moe_utils, token_dispatcher

    original = moe_utils.permute

    def permute(tokens, routing_map, probs=None, num_out_tokens=None, fused=False,
                drop_and_pad=False, tokens_per_expert=None, align_size=0):
        if fused or drop_and_pad or num_out_tokens is None or tokens.dim() != 2:
            return original(tokens, routing_map, probs, num_out_tokens, fused,
                            drop_and_pad, tokens_per_expert, align_size)
        num_tokens = routing_map.shape[0]
        flat_sorted = routing_map.bool().T.contiguous().reshape(-1).argsort(descending=True, stable=True)
        flat_sorted = flat_sorted[:num_out_tokens]
        sorted_indices = flat_sorted % num_tokens
        permuted_probs = None
        if probs is not None:
            permuted_probs = probs.T.contiguous().reshape(-1)[flat_sorted]
        permuted_input = _PermuteRows.apply(tokens, sorted_indices)
        return permuted_input, permuted_probs, sorted_indices, None, tokens_per_expert

    moe_utils.permute = permute
    token_dispatcher.permute = permute


class _ContiguousGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor):
        return tensor.view_as(tensor)

    @staticmethod
    def backward(ctx, grad):
        return grad.contiguous()


def contiguous_grad(tensor: torch.Tensor) -> torch.Tensor:
    return _ContiguousGrad.apply(tensor)


def _stacked_weight(module) -> torch.Tensor:
    """[experts, out, in] weight whose rows the module's per-expert parameters now view."""
    stacked = module.__dict__.get("_spindle_stacked_weight")
    if stacked is None:
        weights = [getattr(module, f"weight{i}") for i in range(module.num_gemms)]
        stacked = torch.stack([w.detach() for w in weights])
        for i, weight in enumerate(weights):
            weight.data = stacked[i]
        module.__dict__["_spindle_stacked_weight"] = stacked
    return stacked


def _install_expert_grouped_mm() -> None:
    from megatron.core.extensions import transformer_engine as te_ext

    cls = te_ext.TEGroupedLinear
    original = cls.forward

    def forward(self, x, m_splits):
        eligible = (
            x.dtype == torch.bfloat16
            and x.dim() == 2
            and x.size(0) > 0
            and not getattr(self, "use_bias", False)
            and not any(getattr(self, f"weight{i}").requires_grad for i in range(self.num_gemms))
            and not self.config.fp8
            and not getattr(self.config, "fp4", None)
        )
        if not eligible:
            return original(self, x, m_splits)
        stacked = _stacked_weight(self)
        if isinstance(m_splits, torch.Tensor):
            counts = m_splits.to(device=x.device, dtype=torch.int32)
        else:
            counts = torch.tensor(list(m_splits), dtype=torch.int32).pin_memory().to(x.device, non_blocking=True)
        offsets = counts.cumsum(0, dtype=torch.int32)
        out = torch._grouped_mm(x, stacked.transpose(1, 2), offsets)
        return out, None

    cls.forward = forward
