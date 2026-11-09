"""Tinker-compatible client with Modal proxy authentication and waiting progress."""

from __future__ import annotations

import functools
import inspect
import logging
import math
import os
import re
import threading
import time
from typing import Any

import httpx
import tinker
from tinker.lib.api_future_impl import QueueState, QueueStateObserver, _APIFuture

_log = logging.getLogger(__name__)
_adapter_lock = threading.Lock()
_adapter_installed = False


class _Progress(QueueStateObserver):
    def __init__(self, request_id: str, operation: str, interval: float):
        self.request_id = request_id
        self.operation = operation
        self.interval = interval
        self.last_message: tuple[str, str] | None = None
        self.last_report = 0.0

    def on_queue_state_change(self, queue_state, queue_state_reason):
        if queue_state == QueueState.ACTIVE:
            self.last_message = None
            return
        reason = queue_state_reason or f"waiting ({queue_state.value})"
        # Elapsed counters change every poll without representing a new stage.
        stage = re.sub(r"\b\d+[hms](?: \d+[hms])*\b", "<elapsed>", reason)
        message = (queue_state.value, stage)
        now = time.monotonic()
        if message != self.last_message or now - self.last_report >= self.interval:
            _log.warning("%s [%s]: %s", self.operation, self.request_id, reason)
            self.last_message = message
            self.last_report = now


def _install_progress_adapter() -> None:
    global _adapter_installed
    with _adapter_lock:
        if _adapter_installed:
            return
        original = _APIFuture.__init__
        signature = inspect.signature(original)

        @functools.wraps(original)
        def initialize(self, *args, **kwargs):
            bound = signature.bind(self, *args, **kwargs)
            holder = bound.arguments["holder"]
            interval = getattr(holder, "_spindle_progress_interval", None)
            if interval is not None:
                future = bound.arguments["untyped_future"]
                bound.arguments["queue_state_observer"] = _Progress(
                    future.request_id, bound.arguments["request_type"], interval
                )
            original(*bound.args, **bound.kwargs)

        # Tinker has no public observer hook; only Spindle-owned holders opt in.
        _APIFuture.__init__ = initialize
        _adapter_installed = True


class ServiceClient(tinker.ServiceClient):
    """Use Tinker's API with native Modal proxy auth and timely waiting messages.

    Proxy credentials default to MODAL_PROXY_TOKEN_ID / MODAL_PROXY_TOKEN_SECRET.
    Supply both explicit credentials together to override the environment. Custom
    headers take precedence. Set progress=False to retain Tinker's logging, or
    progress_interval to change the interval between repeated waiting messages.
    All other arguments and returned clients/futures follow the installed Tinker SDK.
    """

    def __init__(
        self,
        *args: Any,
        proxy_token_id: str | None = None,
        proxy_token_secret: str | None = None,
        progress: bool = True,
        progress_interval: float = 60.0,
        **kwargs: Any,
    ):
        if not math.isfinite(progress_interval) or progress_interval <= 0:
            raise ValueError("progress_interval must be finite and greater than zero")
        if proxy_token_id is None and proxy_token_secret is None:
            proxy_token_id = os.environ.get("MODAL_PROXY_TOKEN_ID")
            proxy_token_secret = os.environ.get("MODAL_PROXY_TOKEN_SECRET")
        if bool(proxy_token_id) != bool(proxy_token_secret):
            raise ValueError(
                "Set both proxy_token_id and proxy_token_secret, or both MODAL_PROXY_TOKEN_ID and MODAL_PROXY_TOKEN_SECRET"
            )
        headers = httpx.Headers()
        if proxy_token_id and proxy_token_secret:
            headers.update(
                {"Modal-Key": proxy_token_id, "Modal-Secret": proxy_token_secret}
            )
        if "default_headers" in kwargs and "default_headers_overrides" in kwargs:
            raise ValueError(
                "Use only one of default_headers and default_headers_overrides"
            )
        headers.update(
            kwargs.pop("default_headers_overrides", None)
            or kwargs.pop("default_headers", None)
            or {}
        )
        header_arg = (
            "default_headers_overrides"
            if "default_headers_overrides"
            in inspect.signature(tinker.ServiceClient.__init__).parameters
            else "default_headers"
        )
        kwargs[header_arg] = dict(headers)
        self._spindle_progress_interval = progress_interval if progress else None
        if progress:
            _install_progress_adapter()
        super().__init__(*args, **kwargs)

    def _get_session_holder(self):
        holder = super()._get_session_holder()
        holder._spindle_progress_interval = self._spindle_progress_interval
        return holder
