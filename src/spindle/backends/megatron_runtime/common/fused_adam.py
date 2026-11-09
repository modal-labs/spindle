from functools import wraps


_MAX_TENSORS_PER_ADAM_CALL = 1024
_ADAM_CALLBACKS = (
    "multi_tensor_adam",
    "multi_tensor_adam_param_remainder",
    "multi_tensor_adam_fp8",
    "multi_tensor_adam_capturable",
    "multi_tensor_adam_capturable_master",
)


def _bounded_adam_call(function):
    @wraps(function)
    def call(chunk_size, noop_flag, tensor_lists, *args):
        count = len(tensor_lists[0])
        if count <= _MAX_TENSORS_PER_ADAM_CALL:
            return function(chunk_size, noop_flag, tensor_lists, *args)
        for start in range(0, count, _MAX_TENSORS_PER_ADAM_CALL):
            function(
                chunk_size,
                noop_flag,
                [
                    values[start : start + _MAX_TENSORS_PER_ADAM_CALL]
                    for values in tensor_lists
                ],
                *args,
            )

    return call


def bound_fused_adam_tensor_lists(optimizer):
    children = getattr(optimizer, "chained_optimizers", None)
    if children is not None:
        for child in children:
            bound_fused_adam_tensor_lists(child)
        return
    inner = getattr(optimizer, "optimizer", None)
    if inner is not None:
        bound_fused_adam_tensor_lists(inner)
        return
    if not type(optimizer).__module__.startswith(
        "transformer_engine.pytorch.optimizers"
    ):
        return
    if type(optimizer).__name__ != "FusedAdam":
        return
    if (
        max((len(group["params"]) for group in optimizer.param_groups), default=0)
        <= _MAX_TENSORS_PER_ADAM_CALL
    ):
        return
    # TE 2.17's fixed tensor-handle pool cannot hold large MoE Adam tensor lists.
    for name in _ADAM_CALLBACKS:
        callback = getattr(optimizer, name, None)
        if not callable(callback):
            raise RuntimeError(
                f"unsupported Transformer Engine FusedAdam callback: {name}"
            )
        setattr(optimizer, name, _bounded_adam_call(callback))
