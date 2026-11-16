from __future__ import annotations

import functools
from pathlib import Path

import modal

from spindle.usage import DirectorySink, UsageMeter

USAGE_ROOT = "/usage"


def usage_volume(frontend: str) -> modal.Volume:
    return modal.Volume.from_name(
        f"{frontend}-usage", create_if_missing=True, version=2
    )


@functools.cache
def usage_sink(frontend: str) -> DirectorySink:
    """One sink per process, so its lock covers all of this process's Volume I/O."""
    volume = usage_volume(frontend)
    return DirectorySink(USAGE_ROOT, commit=volume.commit, reload=volume.reload)


def start_usage_meter(frontend: str) -> UsageMeter:
    return UsageMeter(usage_sink(frontend)).start()


@functools.cache
def sample_meter(frontend: str) -> UsageMeter | None:
    if not Path(USAGE_ROOT).is_dir():
        return None
    return start_usage_meter(frontend)
