"""NumPy-backed TensorData avoiding Python list materialization across UDS hops."""

from __future__ import annotations

from typing import Any

import numpy as np
from pydantic import SerializerFunctionWrapHandler, model_serializer
from tinker.types._pydantic_types.tensor_data import TensorData

_DECLARED_NP_DTYPES = {"int64": np.int64, "float32": np.float32}


class ArrayTensorData(TensorData):
    """TensorData backed by a contiguous NumPy array until `.data` is accessed."""

    @classmethod
    def from_array(
        cls,
        values: np.ndarray,
        *,
        dtype: str,
        shape: list[int] | None = None,
        sparse_crow_indices: list[int] | None = None,
        sparse_col_indices: list[int] | None = None,
    ) -> ArrayTensorData:
        flat = np.ascontiguousarray(values).reshape(-1)
        instance = cls.model_construct(
            dtype=dtype,
            shape=shape,
            sparse_crow_indices=sparse_crow_indices,
            sparse_col_indices=sparse_col_indices,
        )
        # Defer Python list conversion until `.data` is explicitly read.
        instance.__dict__.pop("data", None)
        object.__setattr__(instance, "_array", flat)
        return instance

    def __getattr__(self, name: str) -> object:
        if name == "data":
            arr = object.__getattribute__(self, "_array")
            values = arr.tolist() if arr is not None else []
            self.__dict__["data"] = values
            # Once materialized as a mutable Python list, drop _array so mutations stay authoritative.
            object.__setattr__(self, "_array", None)
            return values
        return super().__getattr__(name)

    @model_serializer(mode="wrap")
    def _serialize(self, handler: SerializerFunctionWrapHandler) -> dict[str, Any]:
        if "data" not in self.__dict__:
            arr = self.__dict__.get("_array")
            self.__dict__["data"] = arr.tolist() if arr is not None else []
            object.__setattr__(self, "_array", None)
        return handler(self)

    def to_numpy_compact(self) -> np.ndarray:
        """Zero-copy NumPy view preserving compact wire dtype (e.g. int16/int32)."""
        if self.sparse_crow_indices is not None:
            return super().to_numpy()
        arr = self.__dict__.get("_array")
        if arr is None:
            return super().to_numpy()
        return arr.reshape(self.shape) if self.shape is not None else arr

    def to_numpy(self) -> np.ndarray:
        if self.sparse_crow_indices is not None:
            return super().to_numpy()
        arr = self.__dict__.get("_array")
        if arr is None:
            return super().to_numpy()
        target = _DECLARED_NP_DTYPES.get(self.dtype, arr.dtype)
        converted = arr.astype(target, copy=False)
        return converted.reshape(self.shape) if self.shape is not None else converted

    def to_torch(self) -> object:
        if self.sparse_crow_indices is not None or self.__dict__.get("_array") is None:
            return super().to_torch()
        # GPU-only dependency; CPU tests do not install torch.
        import torch

        return torch.from_numpy(np.ascontiguousarray(self.to_numpy()))
