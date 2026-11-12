import asyncio
import struct
import time
from collections import deque

import msgspec
import pytest
import xxhash
import zmq
import zmq.asyncio

from spindle.inference.kv_events import (
    KVCache,
    KVEventSubscriber,
    publisher_config,
)


def stored(keys, tokens, parent=None, adapter="adapter-v1", medium="GPU"):
    return dict(
        type="BlockStored",
        block_hashes=keys,
        token_ids=tokens,
        parent_block_hash=parent,
        block_size=len(tokens) // len(keys),
        lora_id=None,
        cache_salt=None,
        lora_name=adapter,
        medium=medium,
    )


def digest(adapter, tokens):
    h = xxhash.xxh3_128((adapter + "\0").encode())
    for token in tokens:
        h.update(struct.pack("<q", token))
    return h.hexdigest()


def packet(events, ts=1.0):
    # SGLang EventBatch is array-like; events are tagged maps.
    return msgspec.msgpack.encode([ts, events, 0])


def test_kv_prefix_hash_matches_router_request_hash():
    tokens = list(range(128))
    cache = KVCache()
    cache.apply(stored([1, 2], tokens, adapter="adapter-v1"))
    assert set(cache.prefixes) == {
        digest("adapter-v1", tokens[:64]),
        digest("adapter-v1", tokens),
    }


def test_chained_pages_eviction_and_adapter_versions():
    cache = KVCache()
    tokens = list(range(192))
    cache.apply(stored([1, 2], tokens[:128]))
    cache.apply(stored([3], tokens[128:], parent=2))
    assert set(cache.prefixes) == {
        digest("adapter-v1", tokens[:n]) for n in (64, 128, 192)
    }
    cache.apply(stored([4], tokens[:64], adapter="adapter-v2"))
    assert digest("adapter-v2", tokens[:64]) in cache.prefixes
    cache.apply(dict(type="BlockRemoved", block_hashes=[2], medium="CPU_PINNED"))
    assert 3 in cache.blocks
    cache.apply(dict(type="BlockRemoved", block_hashes=[2], medium="GPU"))
    assert set(cache.blocks) == {1, 4}
    assert digest("adapter-v1", tokens[:192]) not in cache.prefixes
    cache.apply(dict(type="AllBlocksCleared"))
    assert not cache.prefixes
    assert not cache.children


def test_small_pages_cross_router_boundary_and_unknown_parent():
    cache = KVCache()
    cache.apply(stored(list(range(1, 81)), list(range(80))))
    assert set(cache.prefixes) == {digest("adapter-v1", list(range(64)))}
    cache.apply(stored([100], list(range(64)), parent=999))
    assert 100 not in cache.blocks
    cache.apply(stored([101], list(range(64)), medium="CPU_PINNED"))
    assert 101 not in cache.blocks


def test_duplicate_gap_and_unsalted_events():
    sub = KVEventSubscriber()
    first = packet([stored([1], list(range(64)))])
    sub.apply(0, first)
    sub.apply(0, first)
    sub.ready = True
    assert len(sub.cache.prefixes) == 1
    sub.apply(2, packet([stored([3], list(range(64, 128)), parent=1)]))
    assert not sub.cache.prefixes
    assert not sub.ready
    event = stored([4], list(range(64)))
    sub.cache.apply(event)
    event["cache_salt"] = "tenant-salt"
    sub.cache.apply(event)
    assert set(sub.cache.blocks) == {4}
    assert len(sub.cache.prefixes) == 1


@pytest.mark.parametrize(
    "invalid_event",
    [
        {"lora_name": None, "cache_salt": None, "token_ids": list(range(64))},
        {
            "lora_name": "adapter-v1",
            "cache_salt": "tenant",
            "token_ids": list(range(64)),
        },
        {
            "lora_name": "adapter-v1",
            "cache_salt": None,
            "token_ids": [(i, i + 1) for i in range(64)],
        },
    ],
    ids=["unnamed", "salted", "bigram"],
)
def test_events_without_comparable_named_unigram_prefixes_are_skipped(invalid_event):
    cache = KVCache()
    event = stored([1], list(range(64)))
    event.update(invalid_event)
    cache.apply(event)
    assert not cache.blocks
    assert not cache.prefixes


def test_routing_prefixes_are_bounded_and_favor_new_paths():
    cache = KVCache()
    for key in range(8193):
        cache.apply(stored([key], list(range(key, key + 64))))
    prefixes = cache.prefixes
    assert len(prefixes) == 8192
    assert digest("adapter-v1", list(range(8192, 8256))) in prefixes
    assert digest("adapter-v1", list(range(64))) not in prefixes


def test_publisher_config_has_single_rank_endpoints():
    assert publisher_config()["endpoint"] == "tcp://127.0.0.1:5557"
    assert publisher_config()["replay_endpoint"] == "tcp://127.0.0.1:5558"
    assert publisher_config()["bind"] is True


@pytest.mark.parametrize(
    "with_topic", [False, True], ids=["sglang-0.5.21", "sglang-0.5.23"]
)
def test_replay_uses_sglang_router_framing_and_live_sequence_detects_restarts(
    tmp_path, with_topic
):
    async def scenario():
        context = zmq.asyncio.Context()
        router = context.socket(zmq.ROUTER)
        endpoint = f"ipc://{tmp_path}/replay"
        router.bind(endpoint)
        subscriber = KVEventSubscriber(replay_endpoint=endpoint)
        history = [
            (0, packet([stored([1], list(range(64)))])),
            (1, packet([stored([2], list(range(64, 128)), parent=1)])),
        ]

        async def serve(requests=1):
            for _ in range(requests):
                identity, delimiter, start = await router.recv_multipart()
                assert delimiter == b""
                start = int.from_bytes(start, "big")
                for seq, payload in history:
                    if seq >= start:
                        prefix = (
                            [identity, b"", b"kv-events"]
                            if with_topic
                            else [identity, b""]
                        )
                        await router.send_multipart(
                            prefix + [seq.to_bytes(8, "big"), payload]
                        )
                prefix = [identity, b"", b""] if with_topic else [identity, b""]
                await router.send_multipart(
                    prefix + [(-1).to_bytes(8, "big", signed=True), b""]
                )

        async def replay(requests=1):
            await asyncio.gather(serve(requests), subscriber.replay(context))

        try:
            await replay()
            assert subscriber.ready
            assert subscriber.sequence == 1
            assert len(subscriber.cache.prefixes) == 2
            await replay()  # Anchor is deduplicated, including a quiet stream.
            assert len(subscriber.cache.prefixes) == 2
            history[:] = [(0, packet([stored([9], list(range(64)))], ts=2.0))]
            await replay(2)  # A missing anchor probes the replay buffer from zero.
            assert subscriber.ready
            assert subscriber.sequence == 0
            assert set(subscriber.cache.blocks) == {9}
            assert digest("adapter-v1", list(range(64))) in subscriber.cache.prefixes
        finally:
            router.close(0)
            context.term()

    asyncio.run(asyncio.wait_for(scenario(), 5))


@pytest.mark.parametrize(
    "with_topic", [False, True], ids=["sglang-0.5.21", "sglang-0.5.23"]
)
def test_replay_recovers_after_buffer_roll_without_inventing_ancestors(
    tmp_path, with_topic
):
    async def scenario():
        context = zmq.asyncio.Context()
        router = context.socket(zmq.ROUTER)
        endpoint = f"ipc://{tmp_path}/rolled"
        router.bind(endpoint)
        subscriber = KVEventSubscriber(replay_endpoint=endpoint)
        history = deque(maxlen=2)

        def add(sequence, event):
            history.append((sequence, packet([event])))

        add(0, stored([1], list(range(64))))
        add(1, stored([2], list(range(64, 128)), parent=1))
        add(2, stored([3], list(range(128, 192)), parent=2))
        add(3, stored([4], list(range(64))))
        add(4, stored([5], list(range(64, 128)), parent=4))

        async def serve():
            identity, delimiter, _ = await router.recv_multipart()
            assert delimiter == b""
            for seq, payload in history:
                prefix = (
                    [identity, b"", b"kv-events"] if with_topic else [identity, b""]
                )
                await router.send_multipart(prefix + [seq.to_bytes(8, "big"), payload])
            prefix = [identity, b"", b""] if with_topic else [identity, b""]
            await router.send_multipart(
                prefix + [(-1).to_bytes(8, "big", signed=True), b""]
            )

        try:
            await asyncio.gather(serve(), subscriber.replay(context))
            assert subscriber.ready
            assert subscriber.sequence == 4
            assert set(subscriber.cache.blocks) == {4, 5}
            assert set(subscriber.cache.prefixes) == {
                digest("adapter-v1", list(range(64))),
                digest("adapter-v1", list(range(128))),
            }
        finally:
            router.close(0)
            context.term()

    asyncio.run(asyncio.wait_for(scenario(), 5))


def test_live_stream_and_idle_replay_recover_dropped_eviction(tmp_path):
    async def scenario():
        context = zmq.asyncio.Context()
        pub = context.socket(zmq.PUB)
        router = context.socket(zmq.ROUTER)
        endpoint = f"ipc://{tmp_path}/pub"
        replay = f"ipc://{tmp_path}/replay"
        pub.bind(endpoint)
        router.bind(replay)
        history = []
        subscriber = KVEventSubscriber(endpoint, replay)

        async def serve():
            while True:
                identity, _, start = await router.recv_multipart()
                for seq, payload in history:
                    if seq >= int.from_bytes(start, "big"):
                        await router.send_multipart(
                            [identity, b"", seq.to_bytes(8, "big"), payload]
                        )
                await router.send_multipart([identity, b"", b"\xff" * 8, b""])

        async def until(predicate):
            async with asyncio.timeout(4):
                while not predicate():
                    await asyncio.sleep(0.01)

        tasks = [asyncio.create_task(serve()), asyncio.create_task(subscriber.run())]
        try:
            await until(lambda: subscriber.ready)
            payload = packet([stored([1], list(range(64)))])
            history.append((0, payload))
            await pub.send_multipart([b"kv-events", (0).to_bytes(8, "big"), payload])
            await until(lambda: len(subscriber.cache.prefixes) == 1)
            history.append((1, packet([stored([2], list(range(64, 128)), parent=1)])))
            history.append(
                (2, packet([dict(type="BlockRemoved", block_hashes=[1], medium="GPU")]))
            )
            await pub.send_multipart(
                [b"kv-events", (2).to_bytes(8, "big"), history[-1][1]]
            )
            await until(lambda: subscriber.sequence == 2)
            assert not subscriber.cache.prefixes
            payload = packet([stored([3], list(range(64)))])
            history.append((3, payload))
            await pub.send_multipart([b"kv-events", (3).to_bytes(8, "big"), payload])
            await until(lambda: len(subscriber.cache.prefixes) == 1)
            # No PUB notification: idle replay must recover the final removal.
            history.append(
                (4, packet([dict(type="BlockRemoved", block_hashes=[3], medium="GPU")]))
            )
            await until(lambda: subscriber.sequence == 4)
            assert not subscriber.cache.prefixes
        finally:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            pub.close(0)
            router.close(0)
            context.term()

    asyncio.run(scenario())


def test_live_publications_queued_during_replay_keep_cache(tmp_path):
    async def scenario():
        context = zmq.asyncio.Context()
        pub = context.socket(zmq.PUB)
        router = context.socket(zmq.ROUTER)
        endpoint = f"ipc://{tmp_path}/queued-pub"
        replay = f"ipc://{tmp_path}/queued-replay"
        pub.bind(endpoint)
        router.bind(replay)
        subscriber = KVEventSubscriber(endpoint, replay)
        seen = []
        original = subscriber.apply_publication

        def observe(sequence, payload):
            seen.append((sequence, subscriber.sequence))
            return original(sequence, payload)

        subscriber.apply_publication = observe
        stored_event = stored([1], list(range(64)))
        batches = [
            packet([stored_event]),
            packet([dict(type="BlockRemoved", block_hashes=[999], medium="GPU")]),
        ]

        async def serve():
            identity, delimiter, start = await router.recv_multipart()
            assert delimiter == b"" and int.from_bytes(start, "big") == 0
            await asyncio.sleep(0.2)
            for sequence, payload in enumerate(batches):
                await pub.send_multipart(
                    [b"kv-events", sequence.to_bytes(8, "big"), payload]
                )
            await asyncio.sleep(0.05)
            for sequence, payload in enumerate(batches):
                await router.send_multipart(
                    [identity, b"", b"kv-events", sequence.to_bytes(8, "big"), payload]
                )
            await router.send_multipart(
                [identity, b"", b"", (-1).to_bytes(8, "big", signed=True), b""]
            )

        task = asyncio.create_task(subscriber.run())
        try:
            await asyncio.wait_for(serve(), 3)
            async with asyncio.timeout(2):
                while not seen:
                    await asyncio.sleep(0.01)
            assert any(sequence < current for sequence, current in seen)
            assert subscriber.ready and subscriber.sequence == 1
            assert set(subscriber.cache.prefixes) == {
                digest("adapter-v1", list(range(64)))
            }
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            pub.close(0)
            router.close(0)
            context.term()

    asyncio.run(scenario())


def test_replay_timeout_applies_to_each_frame(tmp_path):
    async def scenario():
        context = zmq.asyncio.Context()
        router = context.socket(zmq.ROUTER)
        endpoint = f"ipc://{tmp_path}/slow-replay"
        router.bind(endpoint)
        subscriber = KVEventSubscriber(replay_endpoint=endpoint)
        subscriber.ready = True
        subscriber._publish_snapshot()

        async def serve():
            identity, delimiter, start = await router.recv_multipart()
            assert delimiter == b"" and int.from_bytes(start, "big") == 0
            assert not subscriber.snapshot[0]
            for sequence in range(3):
                await asyncio.sleep(0.75)
                event = stored(
                    [sequence + 1],
                    list(range(sequence * 64, (sequence + 1) * 64)),
                    parent=sequence if sequence else None,
                )
                await router.send_multipart(
                    [identity, b"", sequence.to_bytes(8, "big"), packet([event])]
                )
            await asyncio.sleep(0.75)
            await router.send_multipart(
                [identity, b"", (-1).to_bytes(8, "big", signed=True), b""]
            )

        try:
            await asyncio.wait_for(
                asyncio.gather(serve(), subscriber.replay(context)), 5
            )
            assert subscriber.ready and subscriber.sequence == 2
            assert len(subscriber.cache.prefixes) == 3
        finally:
            router.close(0)
            context.term()

    asyncio.run(scenario())


def test_subscriber_thread_recovers_after_unexpected_failure(tmp_path):
    async def scenario():
        context = zmq.asyncio.Context()
        router = context.socket(zmq.ROUTER)
        endpoint = f"ipc://{tmp_path}/thread-replay"
        router.bind(endpoint)

        class FailingOnce(KVEventSubscriber):
            attempts = 0

            async def replay(self, context):
                self.attempts += 1
                if self.attempts == 1:
                    raise RuntimeError("unexpected failure")
                await super().replay(context)

        subscriber = FailingOnce(replay_endpoint=endpoint)

        async def serve():
            identity, _, _ = await router.recv_multipart()
            await router.send_multipart(
                [identity, b"", (-1).to_bytes(8, "big", signed=True), b""]
            )

        task = asyncio.create_task(serve())
        subscriber.start_thread()
        thread = subscriber._thread
        try:
            async with asyncio.timeout(3):
                while subscriber.attempts < 1:
                    await asyncio.sleep(0.01)
                assert not subscriber.snapshot[0]
                await task
                while not subscriber.snapshot[0]:
                    await asyncio.sleep(0.01)
            assert thread.is_alive() and subscriber.attempts >= 2
        finally:
            await asyncio.to_thread(subscriber.stop_thread)
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            router.close(0)
            context.term()
        assert not thread.is_alive() and not subscriber.snapshot[0]

    asyncio.run(scenario())


def test_subscriber_thread_stops_after_replay_socket_error():
    subscriber = KVEventSubscriber(replay_endpoint="invalid://replay")
    subscriber.start_thread()
    thread = subscriber._thread
    started = time.monotonic()
    subscriber.stop_thread()
    assert time.monotonic() - started < 2.5
    assert not thread.is_alive() and not subscriber.snapshot[0]


def test_native_adapter_name_needs_no_cache_salt():
    cache = KVCache()
    event = stored([1], list(range(64)))
    event["cache_salt"] = None
    event["lora_name"] = "adapter-v1"
    cache.apply(event)
    assert digest("adapter-v1", list(range(64))) in cache.prefixes
    assert cache.adapter_names_seen


def test_cache_keeps_shared_root_with_many_blocks():
    cache = KVCache()
    cache.apply(stored([1], list(range(64))))
    cache.apply(stored([2], list(range(64, 128)), parent=1))
    cache.apply(stored([3], list(range(128, 192)), parent=2))
    assert set(cache.blocks) == {1, 2, 3}
    assert len(cache.prefixes) == 3


@pytest.mark.parametrize("page_size", [1, 16, 64, 80])
def test_store_skips_unknown_parent_and_bigram_tokens(page_size):
    cache = KVCache()
    tokens = list(range(page_size * 2))
    event = stored([1, 2], tokens, parent=999)
    cache.apply(event)
    assert not cache.blocks
    event["parent_block_hash"] = None
    event["token_ids"][0] = (1, 2)
    cache.apply(event)
    cache.apply(stored([3], list(range(page_size)), parent=2))
    assert not cache.blocks
    event["token_ids"][0] = 0
    cache.apply(event)
    cache.apply(stored([3], list(range(page_size)), parent=2))
    cache.apply(dict(type="BlockRemoved", block_hashes=[2], medium="GPU"))
    assert set(cache.blocks) == {1}


def test_repeated_store_and_recent_prefix_cap():
    cache = KVCache()
    keys = list(range(1, 8194))
    tokens = list(range(64 * len(keys)))
    event = stored(keys, tokens)
    cache.apply(event)
    cache.apply(event)
    assert len(cache.blocks) == len(keys)
    assert len(cache.prefixes) == 8192
    assert digest("adapter-v1", tokens) in cache.prefixes
    assert digest("adapter-v1", tokens[:64]) not in cache.prefixes
    cache.apply(stored([8194], tokens[64:128], parent=1))
    assert digest("adapter-v1", tokens[:64]) in cache.prefixes


def test_mixed_restore_extends_existing_prefix():
    cache = KVCache()
    cache.apply(stored([1], list(range(64))))
    cache.apply(stored([1, 2], list(range(128))))
    assert set(cache.blocks) == {1, 2}
    assert set(cache.prefixes) == {
        digest("adapter-v1", list(range(64))),
        digest("adapter-v1", list(range(128))),
    }


def test_signed_block_hashes_survive_resize_and_removal():
    cache = KVCache()
    keys = [-(1 << 63), (1 << 63) - 1] + list(range(1024))
    tokens = list(range(len(keys)))
    cache.apply(stored(keys, tokens))
    cache.apply(dict(type="BlockRemoved", block_hashes=[keys[1]], medium="GPU"))
    assert set(cache.blocks) == {keys[0]}
    assert not cache.prefixes
