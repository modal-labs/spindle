"""Stock Tinker clients with Modal proxy authentication and timely waiting logs."""

from __future__ import annotations

import inspect
import os
from typing import Any

import httpx
import tinker
from tinker.lib import queue_state_logger


def create_service_client(
    *args: Any,
    proxy_token_id: str | None = None,
    proxy_token_secret: str | None = None,
    **kwargs: Any,
) -> tinker.ServiceClient:
    """Return a stock Tinker client with Spindle connection defaults.

    Proxy credentials default to MODAL_PROXY_TOKEN_ID / MODAL_PROXY_TOKEN_SECRET.
    Supply both explicit credentials together to override the environment. Custom
    headers take precedence. All other arguments follow the installed Tinker SDK.

    This disables Tinker's waiting-log grace period for every client in this
    process. Tinker's observers and 60-second logging interval stay unchanged.
    """
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
    if hasattr(queue_state_logger, "PAUSE_GRACE_PERIOD"):
        queue_state_logger.PAUSE_GRACE_PERIOD = 0
    return tinker.ServiceClient(*args, **kwargs)
