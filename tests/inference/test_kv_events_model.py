"""Model-based checks for the SGLang KV publisher's PUB/ROUTER protocol."""

import asyncio
import random
import struct
from collections import deque
from dataclasses import dataclass

import msgspec
import pytest
import xxhash
import zmq
import zmq.asyncio

from spindle.inference.kv_events import KVEventSubscriber


TOPIC = b"kv-events"
END_SEQ = (-1).to_bytes(8, "big", signed=True)


def packet(events, sequence):
    return msgspec.msgpack.encode([float(sequence), events, 0])


@dataclass
class Node:
    parent: int | None
    adapter: str
    salt: str | None
    tokens: tuple[int, ...]


class TreeModel:
    def __init__(self):
        self.nodes: dict[int, Node] = {}
        self.next_key = 1

    def store(self, parent, adapter, salt, pages):
        keys = list(range(self.next_key, self.next_key + len(pages)))
        self.next_key += len(pages)
        path = self.nodes[parent].tokens if parent is not None else ()
        for key, page in zip(keys, pages, strict=True):
            path += tuple(page)
            self.nodes[key] = Node(parent, adapter, salt, path)
            parent = key
        return {
            "type": "BlockStored",
            "block_hashes": keys,
            "parent_block_hash": self.nodes[keys[0]].parent,
            "token_ids": [token for page in pages for token in page],
            "block_size": len(pages[0]),
            "lora_id": None,
            "lora_name": None if adapter == "base" else adapter,
            "cache_salt": salt,
            "medium": "GPU",
        }

    def remove(self, key):
        removed = {key}
        while True:
            children = {
                node_key
                for node_key, node in self.nodes.items()
                if node.parent in removed
            }
            if children <= removed:
                break
            removed |= children
        for node_key in removed:
            self.nodes.pop(node_key)
        return {"type": "BlockRemoved", "block_hashes": [key], "medium": "GPU"}

    def clear(self):
        self.nodes.clear()
        return {"type": "AllBlocksCleared"}

    def prefixes(self):
        result = set()
        for node in self.nodes.values():
            if node.salt is not None or node.adapter == "base":
                continue
            for length in range(64, len(node.tokens) + 1, 64):
                digest = xxhash.xxh3_128((node.adapter + "\0").encode())
                for token in node.tokens[:length]:
                    digest.update(struct.pack("<q", token))
                result.add(digest.hexdigest())
        return result


def random_stream(seed, count=80, *, include_base=True):
    rng = random.Random(seed)
    model = TreeModel()
    stream = []
    for _ in range(count):
        choices = ["store"] * 7 + ["cpu"] + ["clear"]
        if model.nodes:
            choices += ["remove"] * 3
        choice = rng.choice(choices)
        if choice == "clear":
            event = model.clear()
        elif choice == "remove":
            event = model.remove(rng.choice(list(model.nodes)))
        elif choice == "cpu":
            if rng.randrange(2):
                event = {
                    "type": "BlockRemoved",
                    "block_hashes": list(model.nodes)[:2],
                    "medium": "CPU_PINNED",
                }
            else:
                event = {
                    "type": "BlockStored",
                    "block_hashes": [-model.next_key],
                    "parent_block_hash": None,
                    "token_ids": [rng.randrange(1000) for _ in range(64)],
                    "block_size": 64,
                    "lora_id": None,
                    "lora_name": "adapter-a",
                    "medium": "CPU_PINNED",
                }
        else:
            adapters = ["adapter-a", "adapter-b"]
            if include_base:
                adapters.append("base")
            adapter = rng.choice(adapters)
            salt = rng.choice([None, None, None, "tenant-1"])
            parents = [
                key
                for key, node in model.nodes.items()
                if (node.adapter, node.salt) == (adapter, salt)
            ]
            parent = rng.choice(parents) if parents and rng.randrange(4) else None
            page_size = rng.choice([7, 16, 32, 64, 80])
            pages = [
                [rng.randrange(-(2**31), 2**31) for _ in range(page_size)]
                for _ in range(rng.randint(1, 3))
            ]
            event = model.store(parent, adapter, salt, pages)
        stream.append((event, model.prefixes()))
    return stream


@pytest.mark.parametrize("seed", [2, 11, 37, 83])
@pytest.mark.parametrize("include_base", [False, True], ids=["adapters", "with-base"])
def test_random_valid_stream_matches_independent_tree(seed, include_base):
    subscriber = KVEventSubscriber()
    for sequence, (event, truth) in enumerate(
        random_stream(seed, include_base=include_base)
    ):
        assert subscriber.apply(sequence, packet([event], sequence))
        assert set(subscriber.cache.prefixes) == truth, (
            seed,
            include_base,
            sequence,
            event,
        )


@pytest.mark.parametrize("seed", [5, 41])
def test_removed_tree_releases_all_internal_maps(seed):
    model = TreeModel()
    subscriber = KVEventSubscriber()
    rng = random.Random(seed)
    root = model.store(None, "adapter-a", None, [[1] * 64])
    root_key = root["block_hashes"][0]
    events = [root]
    for value in range(15):
        parent = rng.choice(list(model.nodes))
        events.append(model.store(parent, "adapter-a", None, [[value + 2] * 64]))
    for sequence, event in enumerate(events):
        assert subscriber.apply(sequence, packet([event], sequence))
    sequence = len(events)
    leaves = [
        key
        for key in model.nodes
        if key != root_key and not any(n.parent == key for n in model.nodes.values())
    ]
    for key in leaves[:3]:
        event = model.remove(key)
        subscriber.apply(sequence, packet([event], sequence))
        assert set(subscriber.cache.prefixes) == model.prefixes()
        sequence += 1
    event = model.remove(root_key)
    subscriber.apply(sequence, packet([event], sequence))
    assert not subscriber.cache.blocks
    assert not subscriber.cache.children
    assert not subscriber.cache.prefixes


def test_short_final_page_then_child_crosses_next_prefix_boundary():
    model = TreeModel()
    subscriber = KVEventSubscriber()
    first = model.store(None, "adapter-a", None, [[1] * 64])
    short = model.store(first["block_hashes"][0], "adapter-a", None, [[2] * 7])
    last = model.store(short["block_hashes"][0], "adapter-a", None, [[3] * 64])
    for sequence, event in enumerate([first, short, last]):
        assert subscriber.apply(sequence, packet([event], sequence))
        assert set(subscriber.cache.prefixes) <= model.prefixes()
    assert set(subscriber.cache.prefixes) == model.prefixes()
    assert len(model.prefixes()) == 2


class Publisher:
    def __init__(self, context, buffer_size):
        self.context = context
        self.pub = context.socket(zmq.PUB)
        self.router = context.socket(zmq.ROUTER)
        self.endpoint = (
            f"tcp://127.0.0.1:{self.pub.bind_to_random_port('tcp://127.0.0.1')}"
        )
        self.replay_endpoint = (
            f"tcp://127.0.0.1:{self.router.bind_to_random_port('tcp://127.0.0.1')}"
        )
        self.history = deque(maxlen=buffer_size)
        self.sequence = 0
        self.requests = []

    async def serve(self):
        while True:
            identity, delimiter, start = await self.router.recv_multipart()
            assert delimiter == b"" and len(start) == 8
            requested = int.from_bytes(start, "big")
            self.requests.append(requested)
            for sequence, payload in tuple(self.history):
                if sequence >= requested:
                    await self.router.send_multipart(
                        [identity, b"", TOPIC, sequence.to_bytes(8, "big"), payload]
                    )
            await self.router.send_multipart([identity, b"", b"", END_SEQ, b""])

    async def emit(self, events, *, live=True):
        sequence = self.sequence
        self.sequence += 1
        payload = packet(events, sequence)
        self.history.append((sequence, payload))
        if live:
            await self.pub.send_multipart([TOPIC, sequence.to_bytes(8, "big"), payload])
        return sequence

    def restart(self):
        self.sequence = 0
        self.history.clear()

    def close(self):
        self.pub.close(0)
        self.router.close(0)


async def until(predicate, timeout=3):
    async with asyncio.timeout(timeout):
        while not predicate():
            await asyncio.sleep(0.01)


async def run_scenario(buffer_size, scenario):
    context = zmq.asyncio.Context()
    publisher = Publisher(context, buffer_size)
    subscriber = KVEventSubscriber(publisher.endpoint, publisher.replay_endpoint)
    tasks = [
        asyncio.create_task(publisher.serve()),
        asyncio.create_task(subscriber.run()),
    ]
    try:
        await until(lambda: subscriber.ready)
        await scenario(publisher, subscriber)
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        publisher.close()
        context.term()


def test_gap_replay_with_duplicates_and_reordered_publications():
    async def scenario(publisher, subscriber):
        model = TreeModel()
        first = model.store(None, "adapter-a", None, [[1] * 64])
        await publisher.emit([first])
        await until(lambda: subscriber.sequence == 0)
        second = model.store(first["block_hashes"][0], "adapter-a", None, [[2] * 64])
        await publisher.emit([second], live=False)
        third = model.store(second["block_hashes"][0], "adapter-a", None, [[3] * 64])
        await publisher.emit([third])
        await until(lambda: subscriber.sequence == 2)
        assert subscriber.ready
        assert set(subscriber.cache.prefixes) == model.prefixes()
        await publisher.pub.send_multipart(
            [TOPIC, (1).to_bytes(8, "big"), publisher.history[1][1]]
        )
        await publisher.pub.send_multipart(
            [TOPIC, (2).to_bytes(8, "big"), publisher.history[2][1]]
        )
        await asyncio.sleep(0.05)
        assert subscriber.ready
        assert set(subscriber.cache.prefixes) == model.prefixes()

    asyncio.run(run_scenario(8, scenario))


@pytest.mark.parametrize("seed", [13, 29])
def test_random_delivery_gaps_never_add_false_prefixes(seed):
    async def scenario(publisher, subscriber):
        rng = random.Random(seed)
        stream = random_stream(seed, 45, include_base=False)
        truth_by_sequence = [truth for _, truth in stream]
        violations = []

        async def monitor():
            while True:
                if subscriber.ready and 0 <= subscriber.sequence < len(stream):
                    actual = set(subscriber.cache.prefixes)
                    truth = truth_by_sequence[subscriber.sequence]
                    if not actual <= truth:
                        violations.append((subscriber.sequence, actual - truth))
                await asyncio.sleep(0.002)

        task = asyncio.create_task(monitor())
        try:
            for event, _ in stream:
                await publisher.emit([event], live=rng.randrange(3) != 0)
                await asyncio.sleep(0.003)
            await until(
                lambda: subscriber.ready and subscriber.sequence == len(stream) - 1
            )
            assert not violations
            assert set(subscriber.cache.prefixes) == truth_by_sequence[-1]
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    asyncio.run(run_scenario(64, scenario))


def test_buffer_rollover_recovers_at_next_clear():
    async def scenario(publisher, subscriber):
        model = TreeModel()
        first = model.store(None, "adapter-a", None, [[1] * 64])
        await publisher.emit([first])
        await until(lambda: subscriber.sequence == 0)
        for value in range(5):
            event = model.store(None, "adapter-a", None, [[value + 10] * 64])
            await publisher.emit([event], live=False)
        await publisher.emit([model.clear()], live=False)
        final = model.store(None, "adapter-b", None, [[9] * 64])
        await publisher.emit([final])
        await until(lambda: len(publisher.requests) >= 2)
        next_event = model.store(None, "adapter-b", None, [[8] * 64])
        await publisher.emit([next_event])
        await until(lambda: subscriber.ready and subscriber.sequence == 8)
        assert set(subscriber.cache.prefixes) == model.prefixes()

    asyncio.run(run_scenario(3, scenario))


def test_publisher_restart_resets_sequence_and_state():
    async def scenario(publisher, subscriber):
        model = TreeModel()
        for value in range(3):
            await publisher.emit([model.store(None, "adapter-a", None, [[value] * 64])])
        await until(lambda: subscriber.sequence == 2)
        assert set(subscriber.cache.prefixes) <= model.prefixes()
        publisher.restart()
        model.clear()
        await publisher.emit([{"type": "AllBlocksCleared"}])
        await publisher.emit([model.store(None, "adapter-b", None, [[7] * 64])])
        await until(lambda: subscriber.ready and subscriber.sequence == 1)
        assert set(subscriber.cache.prefixes) == model.prefixes()

    asyncio.run(run_scenario(8, scenario))
