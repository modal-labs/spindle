import asyncio
import json
import struct

import msgspec
import pytest
import xxhash
import zmq
import zmq.asyncio

from spindle.inference.kv_events import (
    KVCache,
    KVEventSubscriber,
    cache_salt,
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
        lora_name=adapter,
        cache_salt=None,
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


def test_cache_salt_rewrite_preserves_per_request_batch_salts():
    rewritten = cache_salt("adapter-v1", ["tenant-a", "tenant-b"])
    assert isinstance(rewritten, list)
    assert [
        json.loads(value.removeprefix("spindle-kv:"))[1] for value in rewritten
    ] == [
        "tenant-a",
        "tenant-b",
    ]


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


def test_duplicate_gap_and_salted_events():
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
    event["cache_salt"] = None
    sub.cache.apply(event)
    assert set(sub.cache.blocks) == {4}
    event["cache_salt"] = cache_salt("adapter-v1", "tenant")
    sub.cache.apply(event)
    assert set(sub.cache.blocks) == {4}


def test_dp_publisher_port_ranges_do_not_overlap():
    assert publisher_config()["replay_endpoint"] == "tcp://*:5558"
    with pytest.raises(ValueError, match="one data-parallel rank"):
        publisher_config(4)


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

        async def serve():
            identity, delimiter, start = await router.recv_multipart()
            assert delimiter == b""
            start = int.from_bytes(start, "big")
            for seq, payload in history:
                if seq >= start:
                    prefix = (
                        [identity, b"", b"kv-events"] if with_topic else [identity, b""]
                    )
                    await router.send_multipart(
                        prefix + [seq.to_bytes(8, "big"), payload]
                    )
            prefix = [identity, b"", b""] if with_topic else [identity, b""]
            await router.send_multipart(
                prefix + [(-1).to_bytes(8, "big", signed=True), b""]
            )

        async def replay():
            await asyncio.gather(serve(), subscriber.replay(context))

        try:
            await replay()
            assert subscriber.ready
            assert subscriber.sequence == 1
            assert len(subscriber.cache.prefixes) == 2
            await replay()  # Anchor is deduplicated, including a quiet stream.
            assert len(subscriber.cache.prefixes) == 2
            history[:] = [(0, packet([], ts=2.0))]
            await (
                replay()
            )  # Replay has no latest-sequence marker when the buffer is empty.
            assert subscriber.ready
            assert len(subscriber.cache.prefixes) == 2
            assert not subscriber.apply_publication(0, packet([], ts=2.0))
            assert subscriber.sequence == -1
            assert not subscriber.cache.prefixes
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
            # No PUB notification: idle replay must recover the final removal.
            history.append(
                (1, packet([dict(type="BlockRemoved", block_hashes=[1], medium="GPU")]))
            )
            await until(lambda: subscriber.sequence == 1)
            assert not subscriber.cache.prefixes
        finally:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            pub.close(0)
            router.close(0)
            context.term()

    asyncio.run(scenario())
