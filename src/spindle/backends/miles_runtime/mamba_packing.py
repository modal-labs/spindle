"""Megatron's Mamba mixer restarts its convolution and state at each packed
sequence only when PackedSeqParams carries seq_idx, which it derives from
total_tokens. Miles leaves total_tokens unset, so every row after the first in
a micro-batch continues from the previous row's state.
"""

from __future__ import annotations

import os

ENV_FLAG = "MAMBA_SEQUENCE_RESET"


def enabled() -> bool:
    return os.environ.get(ENV_FLAG) == "1"


def install(megatron_model) -> None:
    original = megatron_model.get_packed_seq_params
    if getattr(original, "__spindle_mamba_packing__", False):
        return

    def get_packed_seq_params(batch, args):
        params = original(batch, args)
        if (
            params is not None
            and params.seq_idx is None
            and enabled()
            # tokens is this rank's slice under context parallelism; not validated there.
            and args.context_parallel_size == 1
        ):
            params.total_tokens = batch["tokens"].size(-1)
            params.__post_init__()
        return params

    get_packed_seq_params.__spindle_mamba_packing__ = True
    megatron_model.get_packed_seq_params = get_packed_seq_params
