from __future__ import annotations

import logging
from collections.abc import Callable, Iterable, Mapping
from concurrent.futures import Future, ThreadPoolExecutor

from spindle.usage import DEFAULT_LABEL_KEYS, DirectorySink, labels_from


class SessionUsageWriter:
    """Session labels and model owners, kept on the Volume since swept sessions are deleted."""

    def __init__(
        self, sink: DirectorySink, label_keys: Iterable[str] = DEFAULT_LABEL_KEYS
    ) -> None:
        self.sink = sink
        self.label_keys = tuple(label_keys)
        # A Volume commit takes most of a second, so requests never wait on one.
        self._writer = ThreadPoolExecutor(1, thread_name_prefix="usage-writer")
        self._models: set[str] = set()

    def put_session(
        self, session_id: str, user_metadata: Mapping[str, object] | None
    ) -> Future[bool]:
        labels = labels_from(user_metadata, self.label_keys)
        return self._writer.submit(
            self._write, self.sink.put_session, session_id, labels
        )

    def model_used(self, model_id: str, session_id: str) -> None:
        if model_id not in self._models:
            self._models.add(model_id)
            self._writer.submit(self._put_model, model_id, session_id)

    def _put_model(self, model_id: str, session_id: str) -> None:
        if not self._write(self.sink.put_model, model_id, session_id):
            # Written again the next time the model is used.
            self._models.discard(model_id)

    def _write(self, write: Callable[..., None], *args: object) -> bool:
        try:
            write(*args)
        except Exception as exc:  # noqa: BLE001 - usage must not block training
            logging.getLogger(__name__).warning(
                "usage %s failed: %s", write.__name__, exc
            )
            return False
        return True
