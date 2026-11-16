"""Incremental recipe events and metrics, including complete final lines."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .runner import Recorder


class EventReader:
    def __init__(self, recorder: Recorder) -> None:
        self.recorder = recorder
        self.offsets: dict[Path, int] = {}
        self.completed = 0

    def read(self, root: Path, *, final: bool = False) -> None:
        for path in sorted(root.rglob("*.jsonl")):
            if "events" not in path.name and "metrics" not in path.name:
                continue
            source = str(path.relative_to(root))
            offset = self.offsets.get(path, 0)
            if path.stat().st_size < offset:
                offset = 0
            with path.open("rb") as stream:
                stream.seek(offset)
                while line := stream.readline():
                    if not final and not line.endswith(b"\n"):
                        break
                    self.offsets[path] = stream.tell()
                    try:
                        value = json.loads(line)
                        if not isinstance(value, dict):
                            raise ValueError("event must be an object")
                    except ValueError:
                        self.recorder.collection_error(source, "Invalid JSON event")
                        continue
                    event = value.get("event", "metrics")
                    self.completed += event in ("update_done", "step_done")
                    labels = {
                        "source": source,
                        "event": event,
                        "step": value.get("update", value.get("step", 0)),
                    }
                    if "client" in value or "job" in value:
                        labels["client"] = value.get("client", value.get("job"))
                    for name in ("model_id", "operation"):
                        if name in value:
                            labels[name] = value[name]
                    if "phase" in value:
                        labels["phase"] = value["phase"]
                    elif "warmup" in str(event):
                        labels["phase"] = "warmup"
                    stamp = value.get("time")
                    if stamp is not None and (
                        not isinstance(stamp, (int, float)) or not math.isfinite(stamp)
                    ):
                        self.recorder.collection_error(
                            source, "Invalid event timestamp; using collection time"
                        )
                        stamp = None
                    self.recorder.record(
                        {"recipe": value}, labels, raw=value, stamp=stamp
                    )
