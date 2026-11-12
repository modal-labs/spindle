"""Access legacy Miles arguments and v0.2.0 immutable runtime configurations."""

from copy import copy

from pydantic import BaseModel


def backend_config(args):
    return args.backend if isinstance(args, BaseModel) else args


def replace_config(args, **updates):
    if isinstance(args, BaseModel):
        return args.model_copy(update=updates)
    result = copy(args)
    for key, value in updates.items():
        setattr(result, key, value)
    return result
