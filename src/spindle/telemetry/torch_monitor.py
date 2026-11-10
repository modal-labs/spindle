"""Forward process-local torch.monitor events to OTLP metrics."""

import logging
import math
import os
import re
import threading

from .metrics import meter_provider

log = logging.getLogger(__name__)
MAX_INSTRUMENTS = 256
_lock = threading.Lock()
_handler = None
_provider = None
_warned = False


def _warn_once():
    global _warned
    if not _warned:
        _warned = True
        log.warning("Torch monitor metrics failed; some measurements were dropped")


def install(*, definition_id=None, component="trainer"):
    """Register at most one handler in this process; leave global OTel state alone."""
    global _handler, _provider
    if os.getenv("SPINDLE_TORCH_MONITOR_METRICS", "true").lower() in {
        "0",
        "false",
        "no",
        "off",
    }:
        return
    with _lock:
        if _handler is not None:
            return
        provider = None
        try:
            provider = meter_provider()
            if provider is None:
                return
            # Torch is optional in control-plane and local CLI installations.
            import torch.distributed as dist
            import torch.monitor as monitor

            meter = provider.get_meter("spindle.torch_monitor")
            instruments = {}
            attributes = {"spindle.component": component}
            definition_id = definition_id or os.getenv("SPINDLE_DEFINITION_ID")
            if definition_id:
                attributes["spindle.definition_id"] = definition_id
            event_lock = threading.Lock()

            def handle(event):
                try:
                    with event_lock:
                        tags = {**attributes, "torch.monitor.event": event.name}
                        rank = (
                            dist.get_rank()
                            if dist.is_initialized()
                            else os.getenv("RANK")
                        )
                        if rank is not None:
                            tags["spindle.rank"] = int(rank)
                        for key, value in event.data.items():
                            if isinstance(value, bool) or not isinstance(
                                value, (int, float)
                            ):
                                continue
                            if not math.isfinite(value):
                                continue
                            name = (
                                "torch.monitor."
                                + re.sub(r"[^a-zA-Z0-9_.-]", "_", key)[:240]
                            )
                            instrument = instruments.get(name)
                            if instrument is None:
                                if len(instruments) >= MAX_INSTRUMENTS:
                                    continue
                                instrument = meter.create_gauge(name)
                                instruments[name] = instrument
                            instrument.set(value, tags)
                except Exception:
                    _warn_once()

            _handler = monitor.register_event_handler(handle)
            _provider = provider
        except Exception:
            _warn_once()
            if provider is not None:
                try:
                    provider.shutdown()
                except Exception:
                    pass
