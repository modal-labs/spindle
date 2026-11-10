from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import (
    ClassVar as _ClassVar,
    Iterable as _Iterable,
    Mapping as _Mapping,
    Optional as _Optional,
    Union as _Union,
)

DESCRIPTOR: _descriptor.FileDescriptor

class RoutingSnapshot(_message.Message):
    __slots__ = ["active", "adapters", "observed_at", "replica", "work"]
    class AdaptersEntry(_message.Message):
        __slots__ = ["key", "value"]
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: int
        def __init__(
            self, key: _Optional[str] = ..., value: _Optional[int] = ...
        ) -> None: ...

    ACTIVE_FIELD_NUMBER: _ClassVar[int]
    ADAPTERS_FIELD_NUMBER: _ClassVar[int]
    OBSERVED_AT_FIELD_NUMBER: _ClassVar[int]
    REPLICA_FIELD_NUMBER: _ClassVar[int]
    WORK_FIELD_NUMBER: _ClassVar[int]
    active: int
    adapters: _containers.ScalarMap[str, int]
    observed_at: float
    replica: str
    work: Work
    def __init__(
        self,
        replica: _Optional[str] = ...,
        active: _Optional[int] = ...,
        adapters: _Optional[_Mapping[str, int]] = ...,
        observed_at: _Optional[float] = ...,
        work: _Optional[_Union[Work, _Mapping]] = ...,
    ) -> None: ...

class WatchRequest(_message.Message):
    __slots__ = []
    def __init__(self) -> None: ...

class Work(_message.Message):
    __slots__ = [
        "age_seconds",
        "cpu_adapters",
        "gpu_adapters",
        "kv_capacity_tokens",
        "kv_used_tokens",
        "pending_adapters",
        "prefixes",
        "request_ids",
        "waiting_requests",
    ]
    AGE_SECONDS_FIELD_NUMBER: _ClassVar[int]
    CPU_ADAPTERS_FIELD_NUMBER: _ClassVar[int]
    GPU_ADAPTERS_FIELD_NUMBER: _ClassVar[int]
    KV_CAPACITY_TOKENS_FIELD_NUMBER: _ClassVar[int]
    KV_USED_TOKENS_FIELD_NUMBER: _ClassVar[int]
    PENDING_ADAPTERS_FIELD_NUMBER: _ClassVar[int]
    PREFIXES_FIELD_NUMBER: _ClassVar[int]
    REQUEST_IDS_FIELD_NUMBER: _ClassVar[int]
    WAITING_REQUESTS_FIELD_NUMBER: _ClassVar[int]
    age_seconds: float
    cpu_adapters: _containers.RepeatedScalarFieldContainer[str]
    gpu_adapters: _containers.RepeatedScalarFieldContainer[str]
    kv_capacity_tokens: int
    kv_used_tokens: int
    pending_adapters: _containers.RepeatedScalarFieldContainer[str]
    prefixes: _containers.RepeatedScalarFieldContainer[str]
    request_ids: _containers.RepeatedScalarFieldContainer[str]
    waiting_requests: int
    def __init__(
        self,
        age_seconds: _Optional[float] = ...,
        prefixes: _Optional[_Iterable[str]] = ...,
        cpu_adapters: _Optional[_Iterable[str]] = ...,
        gpu_adapters: _Optional[_Iterable[str]] = ...,
        request_ids: _Optional[_Iterable[str]] = ...,
        pending_adapters: _Optional[_Iterable[str]] = ...,
        waiting_requests: _Optional[int] = ...,
        kv_used_tokens: _Optional[int] = ...,
        kv_capacity_tokens: _Optional[int] = ...,
    ) -> None: ...
