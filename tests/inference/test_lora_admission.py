import sys
from types import SimpleNamespace

import pytest

from spindle.inference import lora_packing


@pytest.mark.parametrize("cancelled", [None, "a", "d", "blocked"])
@pytest.mark.parametrize("queued", [4, 8])
def test_completed_upload_survives_until_admission_and_cancellation_releases_it(
    monkeypatch: pytest.MonkeyPatch, cancelled: str | None, queued: int
) -> None:
    class Event:
        def __init__(self) -> None:
            self.polls = 0

        def query(self) -> bool:
            self.polls += 1
            return self.polls > 1

    class Loader:
        def __init__(self) -> None:
            self.lora_manager = SimpleNamespace(
                memory_pool=SimpleNamespace(uid_to_buffer_id={None: 0})
            )
            self.lora_to_overlap_load_event = {}
            self.device_module = SimpleNamespace(
                current_stream=lambda: SimpleNamespace(wait_event=lambda event: None)
            )
            self.uploads = 0

        def _drain_completed_overlap_loads(self) -> None:
            raise AssertionError("readiness replacement was not installed")

        def _try_start_overlap_load(self, lora_id: str, running_loras: set) -> bool:
            protected = running_loras | self.lora_to_overlap_load_event.keys()
            if len(protected | {lora_id}) > 4:
                return False
            slots = self.lora_manager.memory_pool.uid_to_buffer_id
            if len(slots) == 4:
                victims = [uid for uid in slots if uid not in protected]
                non_base = [uid for uid in victims if uid is not None]
                if not victims:
                    return False
                del slots[(non_base or victims)[0]]
            slots[lora_id] = 1
            self.lora_to_overlap_load_event[lora_id] = Event()
            self.uploads += 1
            return True

        def try_overlap_load_lora(self, lora_id: str, running_loras: set) -> bool:
            self._drain_completed_overlap_loads()
            if lora_id in self.lora_to_overlap_load_event:
                return False
            if lora_id in self.lora_manager.memory_pool.uid_to_buffer_id:
                return True
            if not self._try_start_overlap_load(lora_id, running_loras):
                return False
            self._drain_completed_overlap_loads()
            return lora_id not in self.lora_to_overlap_load_event

    class Scheduler:
        def __init__(self, loader: Loader) -> None:
            self.lora_overlap_loader = loader
            self.waiting_queue = [
                SimpleNamespace(lora_id=uid, finished=lambda: False)
                for uid in "abcdefgh"[:queued]
            ]
            self.chunked_req = None
            self.blocked = set()

        def can_schedule_lora_req(
            self, req: SimpleNamespace, running_loras: set
        ) -> bool:
            return (
                req.lora_id not in self.blocked
                and self.lora_overlap_loader.try_overlap_load_lora(
                    req.lora_id, running_loras
                )
            )

        def get_next_batch_to_run(
            self, running_batch: SimpleNamespace, last_batch: object
        ) -> SimpleNamespace:
            running = {req.lora_id for req in running_batch.reqs}
            for req in list(self.waiting_queue):
                if self.can_schedule_lora_req(req, running):
                    running.add(req.lora_id)
                    running_batch.reqs.append(req)
                    self.waiting_queue.remove(req)
            return running_batch

    monkeypatch.setitem(
        sys.modules,
        "sglang.srt.distributed",
        SimpleNamespace(get_tp_group=lambda: SimpleNamespace(cpu_group=None)),
    )
    monkeypatch.setitem(
        sys.modules,
        "sglang.srt.lora.lora_overlap_loader",
        SimpleNamespace(LoRAOverlapLoader=Loader),
    )
    monkeypatch.setitem(
        sys.modules,
        "sglang.srt.managers.scheduler",
        SimpleNamespace(Scheduler=Scheduler),
    )
    lora_packing.install_readiness()
    lora_packing.install_admission()
    loader = Loader()
    scheduler = Scheduler(loader)
    batch = scheduler.get_next_batch_to_run(SimpleNamespace(reqs=[]), None)
    assert not batch.reqs
    if cancelled is not None:
        scheduler.waiting_queue = [
            req for req in scheduler.waiting_queue if req.lora_id != cancelled
        ]
        scheduler.waiting_queue.append(
            SimpleNamespace(lora_id="x", finished=lambda: False)
        )
        if cancelled == "blocked":
            scheduler.blocked.add("a")
    expected = {req.lora_id for req in scheduler.waiting_queue}
    if cancelled == "blocked":
        expected.remove("a")
    admitted = set()
    for _ in range(8):
        batch = scheduler.get_next_batch_to_run(batch, None)
        admitted.update(req.lora_id for req in batch.reqs)
        if len(batch.reqs) == 4:
            batch = SimpleNamespace(reqs=[])
    assert admitted == expected
    assert {req.lora_id for req in scheduler.waiting_queue} == (
        {"a"} if cancelled == "blocked" else set()
    )
    assert not loader._spindle_lora_reservations
    assert loader.uploads == (queued if cancelled is None else queued + 1)
