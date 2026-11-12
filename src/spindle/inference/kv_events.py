"""Replica-local SGLang KV event consumer (no GPU imports).

5557 is PUB; 5558 is a ROUTER serving bounded replay, not a SUB socket.
Replay on startup, sequence gaps and periodic idle checks also recovers dropped
final publications. Missing history drops hints rather than inventing residency.
"""

from __future__ import annotations

import asyncio
import bisect
import logging
import mmap
import struct
import sys
import threading
import time
from array import array
from itertools import islice

import msgspec
import xxhash
import zmq
import zmq.asyncio

logger = logging.getLogger(__name__)

TOPIC = b"kv-events"
MAX_PREFIXES = 8192
REPLAY_STALL_TIMEOUT_SECONDS = 2.0
KV_EVENTS_CONFIG = {
    "publisher": "zmq",
    "endpoint": "tcp://127.0.0.1:5557",
    "bind": True,
    "replay_endpoint": "tcp://127.0.0.1:5558",
    "topic": TOPIC.decode(),
}


def publisher_config() -> dict:
    return dict(KV_EVENTS_CONFIG)


class _BlockIndex:
    def __init__(self):
        self.wide = False
        self._allocate(1024)
        self.count = 0
        self.occupied = 0

    def _allocate(self, size):
        width = 8 if self.wide else 4
        self.storage = mmap.mmap(-1, size * (9 + width))
        view = memoryview(self.storage)
        self.keys = view[: size * 8].cast("q")
        self.refs = view[size * 8 : size * (8 + width)].cast("Q" if self.wide else "I")
        self.states = view[size * (8 + width) :]
        view.release()

    def __len__(self):
        return self.count

    def __contains__(self, key):
        return self.get(key) is not None

    def __iter__(self):
        return (self.keys[i] for i, state in enumerate(self.states) if state == 1)

    def get(self, key):
        mask = len(self.states) - 1
        slot = hash(key) & mask
        while self.states[slot]:
            if self.states[slot] == 1 and self.keys[slot] == key:
                return self.refs[slot]
            slot = (slot + 1) & mask
        return None

    def _insert(self, key, ref):
        mask = len(self.states) - 1
        slot = hash(key) & mask
        while self.states[slot] == 1:
            slot = (slot + 1) & mask
        if self.states[slot] == 0:
            self.occupied += 1
        self.states[slot] = 1
        self.keys[slot] = key
        self.refs[slot] = ref
        self.count += 1

    def insert(self, key, ref):
        if (self.occupied + 1) * 10 >= len(self.states) * 7:
            self._resize(
                len(self.states) * 2
                if self.count * 10 >= len(self.states) * 5
                else len(self.states)
            )
        self._insert(key, ref)

    def pop(self, key):
        mask = len(self.states) - 1
        slot = hash(key) & mask
        while self.states[slot]:
            if self.states[slot] == 1 and self.keys[slot] == key:
                self.states[slot] = 2
                self.count -= 1
                return self.refs[slot]
            slot = (slot + 1) & mask
        return None

    def _resize(self, size):
        old_keys, old_refs, old_states = self.keys, self.refs, self.states
        old_storage = self.storage
        self._allocate(size)
        self.count = self.occupied = 0
        for i, state in enumerate(old_states):
            if state == 1:
                self._insert(old_keys[i], old_refs[i])
        old_keys.release()
        old_refs.release()
        old_states.release()
        old_storage.close()

    def widen(self):
        self.wide = True
        self._resize(len(self.states))

    def clear(self):
        self.keys.release()
        self.refs.release()
        self.states.release()
        self.storage.close()
        self.__init__()


class _Chain:
    __slots__ = (
        "id",
        "parent",
        "adapter",
        "length",
        "size",
        "start_digest",
        "storage",
        "count",
        "keys",
        "tokens",
        "positions",
        "digests",
        "prefixes",
    )

    def __init__(self, parent, adapter, length, digest, keys, tokens, size):
        self.id = -1
        self.parent = parent
        self.adapter = adapter
        self.length = length
        self.size = size
        self.start_digest = digest
        self.count = len(keys)
        storage_size = (len(keys) + len(tokens)) * 8
        self.storage = (
            mmap.mmap(-1, storage_size)
            if storage_size >= 4096
            else bytearray(storage_size)
        )
        view = memoryview(self.storage)
        view[: len(keys) * 8] = array("q", keys).tobytes()
        view[len(keys) * 8 :] = tokens.tobytes()
        self.keys = view[: len(keys) * 8].cast("q")
        self.tokens = view[len(keys) * 8 :].cast("q")
        view.release()
        self.positions = array("I")
        self.digests = []
        self.prefixes = []

    def truncate(self, count):
        if count == 0:
            self.keys.release()
            self.tokens.release()
            if isinstance(self.storage, mmap.mmap):
                self.storage.close()
            self.count = 0
            return
        storage_size = count * (self.size + 1) * 8
        storage = (
            mmap.mmap(-1, storage_size)
            if storage_size >= 4096
            else bytearray(storage_size)
        )
        view = memoryview(storage)
        view[: count * 8] = self.keys[:count].cast("B")
        view[count * 8 :] = self.tokens[: count * self.size].cast("B")
        self.keys.release()
        self.tokens.release()
        if isinstance(self.storage, mmap.mmap):
            self.storage.close()
        self.storage = storage
        self.keys = view[: count * 8].cast("q")
        self.tokens = view[count * 8 :].cast("q")
        view.release()
        self.count = count

    def digest_at(self, block):
        end = (block + 1) * self.size
        checkpoint = bisect.bisect_right(self.positions, end) - 1
        if checkpoint >= 0:
            digest = self.digests[checkpoint].copy()
            start = self.positions[checkpoint]
        else:
            digest = self.start_digest.copy()
            start = 0
        if start < end:
            digest.update(memoryview(self.tokens).cast("B")[start * 8 : end * 8])
        return digest


class KVCache:
    def __init__(self):
        self.blocks = _BlockIndex()
        self.chains: list[_Chain | None] = []
        self.free_chains: list[int] = []
        self.children: dict[int, set[int]] = {}
        self.prefix_counts: dict[str, int] = {}
        self.adapter_names_seen = False

    @property
    def prefixes(self):
        return list(islice(reversed(self.prefix_counts), 8192))

    def clear(self):
        self.blocks.clear()
        for chain in self.chains:
            if chain is not None:
                chain.truncate(0)
        self.chains.clear()
        self.free_chains.clear()
        self.children.clear()
        self.prefix_counts.clear()
        self.adapter_names_seen = False

    def _chain_ref(self, key):
        ref = self.blocks.get(key)
        if ref is None:
            return None
        chain = self.chains[ref >> 16]
        return chain, ref & 65535

    def _refresh_ancestors(self, chain, block):
        while chain is not None:
            end = (block + 1) * chain.size
            for position, prefix in zip(chain.positions, chain.prefixes, strict=True):
                if position > end:
                    break
                self.prefix_counts[prefix] = self.prefix_counts.pop(prefix)
            parent = self._chain_ref(chain.parent) if chain.parent is not None else None
            if parent is None:
                break
            chain, block = parent

    def remove(self, key: int):
        found = self._chain_ref(key)
        if found is None:
            return
        pending = [(found[0], found[1])]
        while pending:
            chain, first = pending.pop()
            if not chain.count or first >= chain.count:
                continue
            for child_key in chain.keys[first : chain.count]:
                for child_id in self.children.pop(child_key, ()):
                    child = self.chains[child_id]
                    if child is not None:
                        pending.append((child, 0))
                self.blocks.pop(child_key)
            for position, prefix in zip(chain.positions, chain.prefixes, strict=True):
                if position > first * chain.size:
                    count = self.prefix_counts[prefix] - 1
                    if count:
                        self.prefix_counts[prefix] = count
                    else:
                        del self.prefix_counts[prefix]
            checkpoint = bisect.bisect_right(chain.positions, first * chain.size)
            del chain.positions[checkpoint:]
            del chain.digests[checkpoint:]
            del chain.prefixes[checkpoint:]
            chain.truncate(first)
            if first == 0:
                if chain.parent is not None:
                    siblings = self.children.get(chain.parent)
                    if siblings is not None:
                        siblings.discard(chain.id)
                        if not siblings:
                            del self.children[chain.parent]
                self.chains[chain.id] = None
                self.free_chains.append(chain.id)

    def apply(self, event: dict):
        kind = event["type"]
        if kind == "AllBlocksCleared":
            self.clear()
            return
        # CPU/HiCache placement does not imply a device cache hit.
        if event.get("medium") not in (None, "GPU"):
            return
        if kind == "BlockRemoved":
            for key in event["block_hashes"]:
                self.remove(key)
            return
        if kind != "BlockStored":
            raise ValueError(f"Unknown KV event: {kind}")
        adapter = event.get("lora_name")
        if not adapter or event.get("cache_salt"):
            return
        self.adapter_names_seen = True
        size = event["block_size"]
        tokens = event["token_ids"]
        keys = event["block_hashes"]
        if size <= 0 or len(tokens) != size * len(keys):
            raise ValueError("Invalid KV block size")
        parent = event["parent_block_hash"]
        ancestor = self._chain_ref(parent) if parent is not None else None
        if parent is not None and (ancestor is None or ancestor[0].adapter != adapter):
            return
        if not all(isinstance(token, int) for token in tokens):
            return
        existing = [index for index, key in enumerate(keys) if key in self.blocks]
        if existing:
            start = 0
            for index in existing:
                if start < index:
                    segment = dict(event)
                    segment["block_hashes"] = keys[start:index]
                    segment["token_ids"] = tokens[start * size : index * size]
                    segment["parent_block_hash"] = parent
                    self.apply(segment)
                parent = keys[index]
                start = index + 1
            if start < len(keys):
                segment = dict(event)
                segment["block_hashes"] = keys[start:]
                segment["token_ids"] = tokens[start * size :]
                segment["parent_block_hash"] = parent
                self.apply(segment)
            return
        try:
            packed = array("q", tokens)
        except OverflowError:
            return
        if not keys:
            return
        if sys.byteorder != "little":
            packed.byteswap()
        if ancestor is not None:
            self._refresh_ancestors(*ancestor)
        for start in range(0, len(keys), 65536):
            stop = min(start + 65536, len(keys))
            group_keys = keys[start:stop]
            group_tokens = packed[start * size : stop * size]
            if ancestor is None:
                digest = xxhash.xxh3_128((adapter + "\0").encode())
                length = 0
            else:
                digest = ancestor[0].digest_at(ancestor[1])
                length = ancestor[0].length + (ancestor[1] + 1) * ancestor[0].size
            chain = _Chain(
                parent, adapter, length, digest.copy(), group_keys, group_tokens, size
            )
            chain_id = self.free_chains.pop() if self.free_chains else len(self.chains)
            chain.id = chain_id
            if chain_id >= 65536 and not self.blocks.wide:
                self.blocks.widen()
            if chain_id == len(self.chains):
                self.chains.append(chain)
            else:
                self.chains[chain_id] = chain
            if parent is not None:
                self.children.setdefault(parent, set()).add(chain_id)
            data = memoryview(group_tokens).cast("B")
            offset = 0
            next_boundary = 64 - length % 64
            for index, key in enumerate(group_keys):
                end = (index + 1) * size
                while offset + next_boundary <= end:
                    boundary = offset + next_boundary
                    digest.update(data[offset * 8 : boundary * 8])
                    offset = boundary
                    chain.positions.append(boundary)
                    chain.digests.append(digest.copy())
                    prefix = digest.hexdigest()
                    chain.prefixes.append(prefix)
                    self.prefix_counts[prefix] = self.prefix_counts.pop(prefix, 0) + 1
                    next_boundary = 64
                self.blocks.insert(key, chain_id << 16 | index)
            if offset < len(group_tokens):
                digest.update(data[offset * 8 :])
            parent = group_keys[-1]
            ancestor = (chain, len(group_keys) - 1)


class KVEventSubscriber:
    def __init__(
        self, endpoint="tcp://127.0.0.1:5557", replay_endpoint="tcp://127.0.0.1:5558"
    ):
        self.endpoint = endpoint
        self.replay_endpoint = replay_endpoint
        self.cache = KVCache()
        self.sequence = -1
        self.anchor = None
        self.ready = False
        self.checked_at = 0.0
        self.snapshot: tuple[bool, float, tuple[str, ...]] = (False, 0.0, ())
        self._thread: threading.Thread | None = None
        self._thread_started = threading.Event()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._run_task: asyncio.Task | None = None

    def _publish_snapshot(self):
        prefixes = tuple(self.cache.prefixes) if self.ready else ()
        self.snapshot = (self.ready, self.checked_at, prefixes)

    def start_thread(self):
        self._thread = threading.Thread(target=self._run_thread, daemon=True)
        self._thread.start()
        if not self._thread_started.wait(2) or self._loop is None:
            raise RuntimeError("KV event subscriber thread did not start")

    def _run_thread(self):
        async def worker():
            self._loop = asyncio.get_running_loop()
            self._run_task = asyncio.current_task()
            self._thread_started.set()
            while True:
                try:
                    await self.run()
                except asyncio.CancelledError:
                    raise
                except Exception:
                    logger.exception("KV event subscriber failed; restarting")
                self.reset()
                await asyncio.sleep(0.5)

        try:
            asyncio.run(worker())
        except asyncio.CancelledError:
            pass
        except Exception:
            logger.exception("KV event subscriber thread stopped")
        finally:
            self.ready = False
            self.snapshot = (False, 0.0, ())
            self._loop = None
            self._thread_started.set()

    def stop_thread(self, timeout: float = 2):
        if self._thread is None:
            return
        if self._loop is not None and self._run_task is not None:
            self._loop.call_soon_threadsafe(self._run_task.cancel)
        self._thread.join(timeout)
        if self._thread.is_alive():
            logger.error(
                "KV event subscriber thread did not stop within %.1fs", timeout
            )
        else:
            self._thread = None

    def reset(self):
        self.cache.clear()
        self.sequence = -1
        self.anchor = None
        self.ready = False
        self._publish_snapshot()

    def apply(self, sequence: int, payload: bytes):
        if sequence <= self.sequence:
            return
        if sequence != self.sequence + 1:
            self.reset()
            return False
        batch = msgspec.msgpack.decode(payload)
        for event in batch[1]:
            self.cache.apply(event)
        self.sequence = sequence
        self.anchor = payload
        self.checked_at = time.monotonic()
        self._publish_snapshot()
        return True

    def apply_publication(self, sequence: int, payload: bytes) -> bool:
        if sequence < self.sequence:
            return True
        if sequence == self.sequence:
            return payload == self.anchor
        return self.apply(sequence, payload)

    async def replay(self, context):
        socket = context.socket(zmq.DEALER)
        try:
            socket.setsockopt(zmq.LINGER, 0)
            socket.connect(self.replay_endpoint)
            self.ready = False
            self._publish_snapshot()

            async def fetch(start: int):
                await socket.send_multipart([b"", start.to_bytes(8, "big")])
                batches = []
                while True:
                    async with asyncio.timeout(REPLAY_STALL_TIMEOUT_SECONDS):
                        frames = await socket.recv_multipart()
                    topic = TOPIC
                    if len(frames) == 3:
                        delimiter, seq, payload = frames
                    elif len(frames) == 4:
                        delimiter, topic, seq, payload = frames
                    else:
                        raise ValueError("Invalid KV replay frame count")
                    if delimiter != b"" or len(seq) != 8:
                        raise ValueError("Invalid KV replay frames")
                    sequence = int.from_bytes(seq, "big", signed=True)
                    if sequence == -1:
                        return batches
                    if topic != TOPIC:
                        raise ValueError("Invalid KV replay topic")
                    batches.append((sequence, payload))

            batches = await fetch(max(self.sequence, 0))
            if self.sequence > 0 and not any(
                sequence == self.sequence for sequence, _ in batches
            ):
                batches = await fetch(0)
            anchor_index = next(
                (
                    i
                    for i, (sequence, _) in enumerate(batches)
                    if sequence == self.sequence
                ),
                None,
            )
            if self.sequence >= 0 and (
                anchor_index is None or batches[anchor_index][1] != self.anchor
            ):
                self.reset()
                anchor_index = None
            start = anchor_index + 1 if anchor_index is not None else 0
            previous = self.sequence
            for sequence, payload in batches[start:]:
                if sequence != previous + 1:
                    self.cache.clear()
                    self.sequence = sequence - 1
                    self.anchor = None
                    self.ready = False
                    self._publish_snapshot()
                self.apply(sequence, payload)
                previous = sequence
            if not batches and self.sequence >= 0:
                self.reset()
            self.ready = True
            self.checked_at = time.monotonic()
            self._publish_snapshot()
        finally:
            socket.close(0)

    async def run(self):
        context = zmq.asyncio.Context()
        socket = None
        try:
            socket = context.socket(zmq.SUB)
            socket.setsockopt(zmq.LINGER, 0)
            socket.setsockopt(zmq.SUBSCRIBE, TOPIC)
            socket.connect(self.endpoint)
            replay = True
            next_replay = 0.0
            while True:
                try:
                    if replay or time.monotonic() >= next_replay:
                        try:
                            await self.replay(context)
                        except TimeoutError:
                            self.reset()
                            await asyncio.sleep(0.5)
                            continue
                        replay = False
                        next_replay = time.monotonic() + 1.5
                    async with asyncio.timeout(0.5):
                        topic, seq, payload = await socket.recv_multipart()
                        if topic != TOPIC or len(seq) != 8:
                            raise ValueError("Invalid KV publication frames")
                        sequence = int.from_bytes(seq, "big")
                        if not self.apply_publication(sequence, payload):
                            replay = True
                except TimeoutError:
                    continue
                except (
                    ValueError,
                    TypeError,
                    KeyError,
                    IndexError,
                    struct.error,
                    OverflowError,
                    zmq.ZMQError,
                    msgspec.DecodeError,
                ):
                    logger.exception("KV event subscription failed; discarding hints")
                    self.reset()
                    replay = True
                    await asyncio.sleep(0.5)
        finally:
            self.ready = False
            self.snapshot = (False, 0.0, ())
            if socket is not None:
                socket.close(0)
            context.destroy(linger=0)
