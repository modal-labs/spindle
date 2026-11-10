import asyncio
import shutil
import socket

import grpc
import httpx
import pytest
import uvicorn
from fastapi import FastAPI, Request

from spindle.inference.routing_grpc import RoutingBridge, WATCH_PATH, start_proxy
from spindle.proto.routing_pb2 import RoutingSnapshot, WatchRequest


def unused_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@pytest.mark.skipif(
    shutil.which("nginx") is None, reason="nginx is installed in the sampler image"
)
def test_http_and_grpc_share_the_replica_port():
    async def scenario():
        app = FastAPI()

        @app.post("/generate")
        async def generate(request: Request):
            return {
                "body": await request.json(),
                "pin": request.headers.get("modal-flash-upstream"),
            }

        @app.get("/health")
        async def health():
            return {"ok": True}

        async def read():
            return dict(
                replica="replica-a",
                active=0,
                work=dict(age_seconds=0, prefixes=["a" * 32]),
            )

        http_port, grpc_port, port = unused_port(), unused_port(), unused_port()
        bridge = RoutingBridge(read)
        grpc_server = await bridge.start(f"127.0.0.1:{grpc_port}")
        publisher = asyncio.create_task(bridge.publish())
        http_server = uvicorn.Server(
            uvicorn.Config(app, host="127.0.0.1", port=http_port, log_level="error")
        )
        serving = asyncio.create_task(http_server.serve())
        proxy = directory = None
        try:
            async with asyncio.timeout(5):
                while not http_server.started:
                    await asyncio.sleep(0.01)
            proxy, directory = start_proxy("127.0.0.1", port, http_port, grpc_port)
            async with httpx.AsyncClient(
                base_url=f"http://127.0.0.1:{port}",
                transport=httpx.AsyncHTTPTransport(
                    http1=False, http2=True, trust_env=False
                ),
            ) as client:
                async with asyncio.timeout(5):
                    while True:
                        assert proxy.poll() is None
                        try:
                            assert (await client.get("/health")).status_code == 200
                            break
                        except httpx.ConnectError:
                            await asyncio.sleep(0.01)
                result = await client.post(
                    "/generate",
                    json={"input_ids": [1]},
                    headers={"modal-flash-upstream": "replica-a:8000"},
                )
                assert result.json() == {
                    "body": {"input_ids": [1]},
                    "pin": "replica-a:8000",
                }
            async with grpc.aio.insecure_channel(f"127.0.0.1:{port}") as channel:
                watch = channel.unary_stream(
                    WATCH_PATH,
                    request_serializer=WatchRequest.SerializeToString,
                    response_deserializer=RoutingSnapshot.FromString,
                )
                stream = watch(
                    WatchRequest(),
                    metadata=(("modal-flash-upstream", "replica-a:8000"),),
                    timeout=5,
                )
                first = await stream.read()
                second = await stream.read()
                assert first.replica == second.replica == "replica-a"
                assert list(first.work.prefixes) == ["a" * 32]
                assert second.observed_at > first.observed_at
                stream.cancel()
        finally:
            if proxy is not None:
                proxy.terminate()
                await asyncio.to_thread(proxy.wait, 5)
                directory.cleanup()
            http_server.should_exit = True
            publisher.cancel()
            await asyncio.gather(publisher, return_exceptions=True)
            await grpc_server.stop(0)
            await serving

    asyncio.run(scenario())
