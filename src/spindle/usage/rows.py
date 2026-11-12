from __future__ import annotations

import json
import logging
import math
import os
import uuid
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, fields
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

HOUR_S = 3600
DAY_S = 86_400
DEFAULT_LABEL_KEYS = ("run_id", "team_id")
MAX_LABEL_LENGTH = 256
ROW_NAMESPACE = uuid.UUID("5d0a7c52-6f1e-4b0f-9a57-3f1d6c0b8e21")


@dataclass(frozen=True)
class UsageRow:
    """One minute of one session's usage on one container; `prompt_tokens` includes cached ones."""

    timestamp: float
    model_id: str
    request_uuid: str
    app_name: str
    session_id: str | None = None
    training_model_id: str | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_prompt_tokens: int = 0
    train_tokens: int = 0
    busy_seconds: float = 0.0
    idle_seconds: float = 0.0
    cold_start_seconds: float = 0.0


COUNTERS = (
    "prompt_tokens",
    "completion_tokens",
    "cached_prompt_tokens",
    "train_tokens",
    "busy_seconds",
    "idle_seconds",
    "cold_start_seconds",
)
_FIELDS = {item.name for item in fields(UsageRow)}


def labels_from(
    user_metadata: Mapping[str, object] | None, keys: Iterable[str]
) -> dict[str, str]:
    if not user_metadata:
        return {}
    labels = {
        key: value
        for key in keys
        if isinstance(value := user_metadata.get(key), str)
        and 0 < len(value) <= MAX_LABEL_LENGTH
    }
    if dropped := sorted(user_metadata.keys() & set(keys) - labels.keys()):
        log.warning(
            "Dropped usage labels %s: values must be 1 to %d characters",
            dropped,
            MAX_LABEL_LENGTH,
        )
    return labels


def label_keys_from_env() -> tuple[str, ...]:
    value = os.environ.get("SPINDLE_USAGE_LABEL_KEYS")
    if value is None:
        return DEFAULT_LABEL_KEYS
    return tuple(key.strip() for key in value.split(",") if key.strip())


def row_from(values: Mapping[str, Any]) -> UsageRow:
    return UsageRow(**{k: v for k, v in values.items() if k in _FIELDS})


def read_rows(path: Path) -> list[UsageRow] | None:
    if not path.is_file():
        return None
    rows = []
    for line in path.read_text().splitlines():
        try:
            rows.append(row_from(json.loads(line)))
        except (ValueError, TypeError):
            log.warning("Skipping an unreadable usage row in %s", path)
    return rows


def to_jsonl(rows: Iterable[UsageRow]) -> str:
    return "".join(json.dumps(asdict(row)) + "\n" for row in rows)


def add_counters(totals: dict[str, float], counters: Mapping[str, float]) -> None:
    for name, value in counters.items():
        totals[name] = totals.get(name, 0) + value


def hour_start(timestamp: float) -> float:
    return math.floor(timestamp / HOUR_S) * HOUR_S


def hour_name(timestamp: float) -> str:
    return datetime.fromtimestamp(timestamp, UTC).strftime("%Y-%m-%dT%H")


def parse_hour_name(name: str) -> float | None:
    try:
        return datetime.strptime(name, "%Y-%m-%dT%H").replace(tzinfo=UTC).timestamp()
    except ValueError:
        return None
