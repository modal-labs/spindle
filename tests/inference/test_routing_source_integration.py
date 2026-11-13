"""CPU process tests for the SGLang publisher, sidecar, and Rust router."""

import atexit
import asyncio
import importlib
import importlib.util
import json
import multiprocessing
import os
import socket
import sys
import threading
import time
import types
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import AsyncMock

import httpx
import pytest
import uvicorn
import xxhash

from spindle.inference import lora_sidecar
from spindle.inference.kv_events import KVEventSubscriber

pytest.importorskip("spindle_router")
NativeRouter = importlib.import_module("spindle.inference.native_router").NativeRouter

FORK_MODULE = Path(
    "/tmp/spindle-sglang-fork/python/sglang/srt/disaggregation/kv_events.py"
)
if not FORK_MODULE.is_file():
    pytest.skip("pinned SGLang fork clone is unavailable", allow_module_level=True)

ADAPTER = "adapter/v1"
TOKENS = list(range(128))


def fork_events():
    """Load the fork's real wire and publisher code without its GPU imports."""
    runtime = types.ModuleType("sglang.srt.runtime_context")
    runtime.get_parallel = lambda: None
    network = types.ModuleType("sglang.srt.utils.network")
    network.NetworkAddress = type("NetworkAddress", (), {})
    saved = {
        name: sys.modules.get(name) for name in (runtime.__name__, network.__name__)
    }
    sys.modules[runtime.__name__] = runtime
    sys.modules[network.__name__] = network
    try:
        spec = importlib.util.spec_from_file_location("fork_kv_events", FORK_MODULE)
        assert spec and spec.loader
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        for name, prior in saved.items():
            if prior is None:
                del sys.modules[name]
            else:
                sys.modules[name] = prior


def free_ports(count):
    sockets = [socket.socket() for _ in range(count)]
    try:
        for sock in sockets:
            sock.bind(("127.0.0.1", 0))
        return [sock.getsockname()[1] for sock in sockets]
    finally:
        for sock in sockets:
            sock.close()


def wait_for(check, timeout=5):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            value = check()
            if value:
                return value
        except (httpx.HTTPError, OSError, KeyError):
            pass
        time.sleep(0.05)
    pytest.fail("component did not become ready")


def serve_sidecar(replica, port, upstream, source, catalog, endpoint, replay_endpoint):
    os.environ["MODAL_TASK_ID"] = replica
    lora_sidecar.ROUTING_CATALOG_PATH = Path(catalog)
    app = lora_sidecar.create_app(
        None,
        upstream,
        routing_source=source,
        kv_subscriber=(
            KVEventSubscriber(endpoint, replay_endpoint) if source != "hook" else None
        ),
    )
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="error")


@pytest.fixture
def harness(tmp_path):
    state = {"loads": "ok"}
    destinations = {}

    class Upstream(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def respond(self, value, status=200, replica=None):
            body = json.dumps(value).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            if replica:
                self.send_header("x-spindle-replica", replica)
            self.end_headers()
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def do_GET(self):
            if self.path != "/v1/loads?include=core":
                self.respond({}, 404)
                return
            if state["loads"] == "slow":
                time.sleep(2)
            self.respond(
                {
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "version": "0.5.23",
                    "accelerator": None,
                    "num_accelerators": 0,
                    "loads": [
                        {
                            "timestamp": time.time(),
                            "dp_rank": 0,
                            "num_running_reqs": 0,
                            "num_waiting_reqs": 0,
                            "num_used_tokens": 0,
                            "max_total_num_tokens": 4096,
                        }
                    ],
                }
            )

        def do_POST(self):
            self.rfile.read(int(self.headers["Content-Length"]))
            if self.path == "/generate":
                self.respond({"text": "ok", "meta_info": {}})
                return
            self.respond({}, 404)

    class Gateway(Upstream):
        def proxy(self):
            replica = self.headers.get("Modal-Flash-Upstream", "cold:8000").split(":")[
                0
            ]
            target = destinations[replica]
            body = self.rfile.read(int(self.headers.get("Content-Length", "0")))
            with httpx.Client(timeout=5) as client:
                response = client.request(
                    self.command,
                    target + self.path,
                    content=body,
                    headers={"Content-Type": "application/json"},
                )
            self.send_response(response.status_code)
            self.send_header("Content-Length", str(len(response.content)))
            self.send_header("Content-Type", "application/json")
            for key in ("x-spindle-replica", "x-spindle-replay-state"):
                if key in response.headers:
                    self.send_header(key, response.headers[key])
            self.end_headers()
            self.wfile.write(response.content)

        do_GET = do_POST = proxy

    servers = [
        ThreadingHTTPServer(("127.0.0.1", 0), handler)
        for handler in (Upstream, Gateway)
    ]
    threads = [threading.Thread(target=s.serve_forever, daemon=True) for s in servers]
    for thread in threads:
        thread.start()
    processes = {}
    publishers = []
    events = fork_events()
    endpoints = {}

    class Harness:
        upstream = f"http://127.0.0.1:{servers[0].server_port}"
        gateway = f"http://127.0.0.1:{servers[1].server_port}"

        def __init__(self):
            self.publishers = {}

        def start(self, replica, source, publisher=True):
            port, pub_port, replay_port = free_ports(3)
            endpoint = f"tcp://127.0.0.1:{pub_port}"
            replay = f"tcp://127.0.0.1:{replay_port}"
            catalog = tmp_path / f"{replica}.json"
            destinations[replica] = f"http://127.0.0.1:{port}"
            process = multiprocessing.get_context("spawn").Process(
                target=serve_sidecar,
                args=(replica, port, self.upstream, source, catalog, endpoint, replay),
            )
            process.start()
            processes[replica] = process
            endpoints[replica] = (endpoint, replay)
            wait_for(
                lambda: (
                    httpx.get(destinations[replica] + "/routing-state").status_code
                    == 200
                )
            )
            if publisher and source != "hook":
                self.publisher(replica)
            return replica

        def publisher(self, replica):
            endpoint, replay = endpoints[replica]
            pub = events.ZmqEventPublisher(
                0,
                endpoint=endpoint,
                replay_endpoint=replay,
                topic="kv-events",
                bind=True,
            )
            publishers.append(pub)
            self.publishers[replica] = pub
            return pub

        def restart(self, replica):
            old = processes[replica]
            old.terminate()
            old.join(timeout=3)
            endpoint, replay = endpoints[replica]
            port = int(destinations[replica].rsplit(":", 1)[1])
            process = multiprocessing.get_context("spawn").Process(
                target=serve_sidecar,
                args=(
                    replica,
                    port,
                    self.upstream,
                    "kv_events",
                    tmp_path / f"{replica}.json",
                    endpoint,
                    replay,
                ),
            )
            process.start()
            processes[replica] = process
            wait_for(
                lambda: (
                    httpx.get(destinations[replica] + "/routing-state").status_code
                    == 200
                )
            )

        def set_loads(self, value):
            state["loads"] = value

        def snapshot(self, replica, prefixes):
            (tmp_path / f"{replica}.json").write_text(
                json.dumps(
                    {
                        "timestamp": time.time(),
                        "requests": [],
                        "prefixes": prefixes,
                        "cpu_adapters": [ADAPTER],
                        "gpu_adapters": [ADAPTER],
                        "snapshot_seconds": 0.0,
                        "cache_kind": "RadixCache",
                        "running_requests": 0,
                        "waiting_requests": 0,
                        "kv_used_tokens": 0,
                        "kv_capacity_tokens": 4096,
                    }
                )
            )

        def publish(self, publisher, event):
            publisher.publish(events.KVEventBatch(time.time(), [event]))

        def stored(self):
            return events.BlockStored(
                block_hashes=[101, 102],
                parent_block_hash=None,
                token_ids=TOKENS,
                block_size=64,
                lora_id=None,
                medium="GPU",
                lora_name=ADAPTER,
            )

        def removed(self):
            return events.BlockRemoved(block_hashes=[101, 102], medium="GPU")

        def state(self, replica):
            return httpx.get(destinations[replica] + "/routing-state").json()

        async def route(self):
            router = NativeRouter(
                AsyncMock(
                    return_value=[
                        {"replica": name, "upstream": f"{name}:8000"}
                        for name in destinations
                    ]
                ),
                gateway=self.gateway,
                headers={},
                capacity=24,
                prefill_tokens_per_second=100,
                balance_absolute=4,
                balance_relative=1.25,
                cache_threshold=0.5,
            )
            try:
                return await router.dispatch(
                    ADAPTER,
                    "session",
                    request_id="integration",
                    body={
                        "input_ids": TOKENS + [128],
                        "lora_path": ADAPTER,
                        "sampling_params": {"max_new_tokens": 4},
                    },
                    headers={"Modal-Session-ID": "session"},
                    deadline=asyncio.get_running_loop().time() + 8,
                )
            finally:
                await router.aclose()

    yield Harness()
    for pub in publishers:
        pub.shutdown()
        atexit.unregister(pub.shutdown)
    for process in processes.values():
        process.terminate()
        process.join(timeout=3)
    for server in servers:
        server.shutdown()
        server.server_close()
    for thread in threads:
        thread.join(timeout=3)


def prefix_hashes():
    digest = xxhash.xxh3_128((ADAPTER + "\0").encode())
    prefixes = []
    for token in TOKENS:
        digest.update(token.to_bytes(8, "little", signed=True))
        if (token + 1) % 64 == 0:
            prefixes.append(digest.hexdigest())
    return prefixes


@pytest.mark.parametrize("source", ["kv_events", "hook"])
def test_real_router_prefers_cached_prefix_then_releases_preference(harness, source):
    harness.start("cached", source)
    harness.start("cold", source)
    if source == "kv_events":
        publisher = harness.publishers["cached"]
        harness.publish(publisher, harness.stored())
        wait_for(
            lambda: (
                len((harness.state("cached")["work"] or {}).get("prefixes", [])) == 2
            )
        )
    else:
        harness.snapshot("cached", prefix_hashes())
        harness.snapshot("cold", [])
        wait_for(
            lambda: (
                len((harness.state("cached")["work"] or {}).get("prefixes", [])) == 2
            )
        )
    wait_for(lambda: harness.state("cold")["work"] is not None)
    first = asyncio.run(harness.route())
    assert first.response.status_code == 200
    assert first.replica == "cached"
    assert first.routing["cached_tokens"] == 128
    if source == "kv_events":
        harness.publish(publisher, harness.removed())
    else:
        harness.snapshot("cached", [])
    wait_for(lambda: (harness.state("cached")["work"] or {}).get("prefixes") == [])
    second = asyncio.run(harness.route())
    assert second.response.status_code == 200
    assert second.routing["best_cached_tokens"] == 0
    assert second.replica == "cold"


def test_slow_loads_keeps_routing_state_fast_and_router_falls_back(harness):
    harness.set_loads("slow")
    harness.start("cold", "kv_events")
    started = time.monotonic()
    response = harness.state("cold")
    assert time.monotonic() - started < 0.5
    assert response["work"] is None
    routed = asyncio.run(harness.route())
    assert routed.response.status_code == 200
    assert routed.routing["reason"] == "gateway_fallback"


def test_late_publisher_and_sidecar_restart_replay_cache(harness):
    harness.start("cached", "kv_events", publisher=False)
    harness.start("cold", "kv_events")
    assert (
        harness.state("cached")["work"] is None
        or not harness.state("cached")["work"]["prefixes"]
    )
    publisher = harness.publisher("cached")
    harness.publish(publisher, harness.stored())
    wait_for(
        lambda: len((harness.state("cached")["work"] or {}).get("prefixes", [])) == 2
    )
    wait_for(lambda: harness.state("cold")["work"] is not None)
    assert asyncio.run(harness.route()).replica == "cached"
    harness.restart("cached")
    wait_for(
        lambda: len((harness.state("cached")["work"] or {}).get("prefixes", [])) == 2
    )
    assert asyncio.run(harness.route()).replica == "cached"
