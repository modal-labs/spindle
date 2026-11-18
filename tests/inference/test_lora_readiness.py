import multiprocessing
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist

from spindle.inference.lora_packing import drain_completed_uploads


def _readiness_rank(rank: int, rendezvous: str) -> None:
    dist.init_process_group(
        "gloo",
        init_method=f"file://{rendezvous}",
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=15),
    )
    try:

        class Event:
            def __init__(self, name: str, ready: bool) -> None:
                self.name = name
                self.ready = ready

            def query(self) -> bool:
                return self.ready

        events = {
            None: Event("base", True),
            "a": Event("a", True),
            "b": Event("b", rank == 0),
        }
        if rank:
            events = dict(reversed(list(events.items())))
        waited = []
        loader = SimpleNamespace(
            lora_to_overlap_load_event=events,
            device_module=SimpleNamespace(
                current_stream=lambda: SimpleNamespace(wait_event=waited.append)
            ),
        )
        torch.set_default_device("meta")
        drain_completed_uploads(loader, None)
        assert set(events) == {"b"}
        assert [event.name for event in waited] == ["base", "a"]
        events["b"].ready = True
        drain_completed_uploads(loader, None)
        assert not events
        assert [event.name for event in waited] == ["base", "a", "b"]
    finally:
        torch.set_default_device("cpu")
        dist.destroy_process_group()


def test_readiness_waits_for_every_rank_with_different_event_order(
    tmp_path: Path,
) -> None:
    context = multiprocessing.get_context("spawn")
    processes = [
        context.Process(target=_readiness_rank, args=(rank, str(tmp_path / "gloo")))
        for rank in range(2)
    ]
    try:
        for process in processes:
            process.start()
        for process in processes:
            process.join(timeout=25)
        assert [process.exitcode for process in processes] == [0, 0]
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
            process.join(timeout=5)
