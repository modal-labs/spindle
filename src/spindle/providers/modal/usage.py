"""Where a deployment's usage goes: the ingest service if configured, else one Volume per frontend."""

from __future__ import annotations

import functools
import os
from pathlib import Path

import modal

from spindle.usage import DirectorySink, UsageMeter
from spindle.usage_rpc import RpcUsageStore

USAGE_ROOT = "/usage"


def usage_volume(frontend: str) -> modal.Volume:
    return modal.Volume.from_name(
        f"{frontend}-usage", create_if_missing=True, version=2
    )


def usage_sink(frontend: str) -> DirectorySink:
    volume = usage_volume(frontend)
    return DirectorySink(USAGE_ROOT, commit=volume.commit, reload=volume.reload)


def usage_store(frontend: str) -> DirectorySink | RpcUsageStore:
    """The usage ingest service when SPINDLE_USAGE_RPC_URL is set, else the Volume."""
    url = os.environ.get("SPINDLE_USAGE_RPC_URL")
    if url:
        return RpcUsageStore(url, deployment=frontend)
    return usage_sink(frontend)


def start_usage_meter(frontend: str) -> UsageMeter:
    return UsageMeter(usage_store(frontend), deployment=frontend).start()


@functools.cache
def sample_meter(frontend: str) -> UsageMeter | None:
    """One meter per sampling container, or none where the Volume is not mounted."""
    if not (os.environ.get("SPINDLE_USAGE_RPC_URL") or Path(USAGE_ROOT).is_dir()):
        return None
    return start_usage_meter(frontend)
