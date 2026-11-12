"""Replica-local SGLang KV event consumer (no GPU imports).

5557 is PUB; 5558 is a ROUTER serving bounded replay, not a SUB socket.
Replay on startup, sequence gaps and periodic idle checks also recovers dropped
final publications. Missing history drops hints rather than inventing residency.
"""

from __future__ import annotations

import asyncio
import json
import logging
import struct
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Protocol

import msgspec
import xxhash
import zmq
import zmq.asyncio

logger = logging.getLogger(__name__)


class Digest(Protocol):
    def copy(self) -> Digest: ...

    def update(self, data: bytes) -> None: ...

    def hexdigest(self) -> str: ...


TOPIC = b"kv-events"
SALT_PREFIX = "spindle-kv:"
MAX_BLOCKS = 100_000
KV_EVENTS_CONFIG = {
    "publisher": "zmq",
    "endpoint": "tcp://*:5557",
    "replay_endpoint": "tcp://*:5558",
    "topic": TOPIC.decode(),
}


def publisher_config(dp_size: int = 1) -> dict:
    config = dict(KV_EVENTS_CONFIG)
    if dp_size != 1:
        raise ValueError("KV cache routing requires one data-parallel rank")
    return config


def cache_salt(
    adapter: str, original: str | list[str] | None = None
) -> str | list[str]:
    if isinstance(original, list):
        return [cache_salt(adapter, item) for item in original]
    # Native lora_id is None; cache salt preserves adapter identity across replay.
    return SALT_PREFIX + json.dumps([adapter, original], separators=(",", ":"))


@dataclass
class Block:
    parent: int | None
    adapter: str
    length: int
    digest: Digest
    prefixes: list[str]


class KVCache:
    def __init__(self):
        self.blocks: dict[int, Block] = {}
        self.children: dict[int, set[int]] = defaultdict(set)
        self.prefixes: Counter[str] = Counter()
        self.adapter_names_seen = False

    def clear(self):
        self.blocks.clear()
        self.children.clear()
        self.prefixes.clear()
        self.adapter_names_seen = False

    def remove(self, key: int):
        pending = [key]
        while pending:
            key = pending.pop()
            pending.extend(self.children.pop(key, ()))
            block = self.blocks.pop(key, None)
            if block is None:
                continue
            if block.parent is not None:
                self.children[block.parent].discard(key)
                if not self.children[block.parent]:
                    del self.children[block.parent]
            for prefix in block.prefixes:
                self.prefixes[prefix] -= 1
                if not self.prefixes[prefix]:
                    del self.prefixes[prefix]

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
        salt = event.get("cache_salt") or ""
        if salt.startswith(SALT_PREFIX):
            salted_adapter, original_salt = json.loads(salt[len(SALT_PREFIX) :])
            if original_salt:
                return
            adapter = adapter or salted_adapter
        elif adapter:
            self.adapter_names_seen = True
            if salt:
                return
        else:
            return
        if event.get("lora_name"):
            self.adapter_names_seen = True
        size = event["block_size"]
        tokens = event["token_ids"]
        keys = event["block_hashes"]
        if size <= 0 or len(tokens) != size * len(keys):
            raise ValueError("Invalid KV block size")
        parent = event["parent_block_hash"]
        for index, key in enumerate(keys):
            ancestor = self.blocks.get(parent)
            if parent is not None and ancestor is None:
                raise ValueError("KV block parent is unknown")
            if ancestor is not None and ancestor.adapter != adapter:
                raise ValueError("KV block crosses adapter namespaces")
            digest = (
                ancestor.digest.copy()
                if ancestor
                else xxhash.xxh3_128((adapter + "\0").encode())
            )
            length = ancestor.length if ancestor else 0
            prefixes = []
            for token in tokens[index * size : (index + 1) * size]:
                digest.update(struct.pack("<q", token))
                length += 1
                if length % 64 == 0:
                    prefixes.append(digest.hexdigest())
            if key not in self.blocks:
                self.blocks[key] = Block(parent, adapter, length, digest, prefixes)
                self.prefixes.update(prefixes)
                if parent is not None:
                    self.children[parent].add(key)
            parent = key
        while len(self.blocks) > MAX_BLOCKS:
            self.remove(next(iter(self.blocks)))


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

    def reset(self):
        self.cache.clear()
        self.sequence = -1
        self.anchor = None
        self.ready = False

    def apply(self, sequence: int, payload: bytes):
        if sequence <= self.sequence:
            return
        if sequence != self.sequence + 1:
            self.cache.clear()
            self.ready = False
            return False
        batch = msgspec.msgpack.decode(payload)
        try:
            for event in batch[1]:
                self.cache.apply(event)
        except (KeyError, TypeError, ValueError, IndexError):
            self.cache.clear()
            raise
        self.sequence = sequence
        self.anchor = payload
        self.checked_at = time.monotonic()
        return True

    def apply_publication(self, sequence: int, payload: bytes) -> bool:
        if sequence < self.sequence:
            self.reset()
            return False
        if sequence == self.sequence:
            if payload != self.anchor:
                self.reset()
                return False
            return True
        return self.apply(sequence, payload)

    async def replay(self, context):
        socket = context.socket(zmq.DEALER)
        socket.setsockopt(zmq.LINGER, 0)
        socket.connect(self.replay_endpoint)
        try:
            await socket.send_multipart(
                [b"", max(self.sequence + 1, 0).to_bytes(8, "big")]
            )
            batches = []
            async with asyncio.timeout(2):
                while True:
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
                        break
                    if topic != TOPIC:
                        raise ValueError("Invalid KV replay topic")
                    batches.append((sequence, payload))
            if (
                batches
                and self.sequence >= 0
                and batches[0][0] == 0
                and self.sequence > 0
            ):
                self.reset()
            if batches and batches[0][0] > self.sequence + 1:
                self.reset()
                return
            for sequence, payload in batches:
                if sequence <= self.sequence:
                    continue
                if not self.apply(sequence, payload):
                    self.reset()
                    return
            self.ready = True
            self.checked_at = time.monotonic()
        finally:
            socket.close()

    async def run(self):
        context = zmq.asyncio.Context()
        socket = context.socket(zmq.SUB)
        socket.setsockopt(zmq.LINGER, 0)
        socket.setsockopt(zmq.SUBSCRIBE, TOPIC)
        socket.connect(self.endpoint)
        replay = True
        next_replay = 0.0
        try:
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
            socket.close()
            context.term()
