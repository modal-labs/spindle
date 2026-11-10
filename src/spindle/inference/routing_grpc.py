"""Native KV/load observations over a replica-pinned gRPC server stream."""

import asyncio
from pathlib import Path
import subprocess
import tempfile
import time

import grpc

from spindle.proto.routing_pb2 import RoutingSnapshot, WatchRequest, Work

SERVICE = "spindle.routing.v1.RoutingObserver"
WATCH_PATH = f"/{SERVICE}/Watch"


def observation(row: dict) -> RoutingSnapshot:
    snapshot = RoutingSnapshot(
        observed_at=time.time(),
        replica=row["replica"],
        active=row["active"],
        adapters=row.get("adapters", {}),
    )
    work = row.get("work")
    if work is not None:
        fields = Work.DESCRIPTOR.fields_by_name
        snapshot.work.CopyFrom(Work(**{k: v for k, v in work.items() if k in fields}))
    return snapshot


class RoutingBridge:
    def __init__(self, read):
        self.read = read
        self.snapshot = None
        self.version = 0
        self.changed = asyncio.Condition()

    async def publish(self):
        while True:
            snapshot = observation(await self.read())
            async with self.changed:
                self.snapshot = snapshot
                self.version += 1
                self.changed.notify_all()
            await asyncio.sleep(0.5)

    async def watch(self, request, context):
        version = -1
        # Rotate streams so gateways can drain and discovery can replace replicas.
        deadline = asyncio.get_running_loop().time() + 30
        while asyncio.get_running_loop().time() < deadline:
            async with self.changed:
                await self.changed.wait_for(
                    lambda: self.snapshot is not None and self.version != version
                )
                snapshot, version = self.snapshot, self.version
            yield snapshot

    async def start(self, address):
        server = grpc.aio.server(
            options=[("grpc.max_send_message_length", 64 * 1024 * 1024)]
        )
        server.add_generic_rpc_handlers(
            (
                grpc.method_handlers_generic_handler(
                    SERVICE,
                    {
                        "Watch": grpc.unary_stream_rpc_method_handler(
                            self.watch,
                            request_deserializer=WatchRequest.FromString,
                            response_serializer=RoutingSnapshot.SerializeToString,
                        ),
                    },
                ),
            )
        )
        if server.add_insecure_port(address) == 0:
            raise RuntimeError(f"Cannot bind routing gRPC server at {address}")
        await server.start()
        return server


def start_proxy(host: str, port: int, http_port: int, grpc_port: int):
    """HTTP and gRPC share Modal's single exposed, authenticated Server port."""
    directory = tempfile.TemporaryDirectory(prefix="spindle-sidecar-")
    Path(directory.name).chmod(0o755)
    config = Path(directory.name) / "nginx.conf"
    listen = f"[{host}]" if ":" in host else host
    config.write_text(f"""
worker_processes 1;
pid {directory.name}/nginx.pid;
error_log stderr warn;
events {{ worker_connections 4096; }}
http {{
    access_log off;
    client_max_body_size 0;
    client_body_temp_path {directory.name}/body;
    proxy_temp_path {directory.name}/proxy;
    fastcgi_temp_path {directory.name}/fastcgi;
    uwsgi_temp_path {directory.name}/uwsgi;
    scgi_temp_path {directory.name}/scgi;
    server {{
        listen {listen}:{port} http2;
        location = {WATCH_PATH} {{
            grpc_pass grpc://127.0.0.1:{grpc_port};
            grpc_read_timeout 35s;
        }}
        location / {{
            proxy_pass http://127.0.0.1:{http_port};
            proxy_http_version 1.1;
            proxy_set_header Connection "";
            proxy_buffering off;
            proxy_read_timeout 3000s;
        }}
    }}
}}
""")
    try:
        process = subprocess.Popen(
            ["nginx", "-c", str(config), "-g", "daemon off;"], start_new_session=True
        )
    except BaseException:
        directory.cleanup()
        raise
    return process, directory
