"""Opt-in benchmark metrics, native Prometheus samples, and device observations."""

from __future__ import annotations

import atexit
import functools
import hashlib
import importlib.metadata
import json
import logging
import math
import os
import re
import socket
import subprocess
import threading

import httpx
from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.sdk.resources import Resource

logger = logging.getLogger(__name__)
SAMPLE = re.compile(
    r"^([a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{(.*)\})?\s+([^\s]+)(?:\s+[^\s]+)?(?:\s+#.*)?$"
)
LABEL = re.compile(r'([a-zA-Z_][a-zA-Z0-9_]*)="((?:[^"\\]|\\.)*)"(?:,\s*|$)')


def enabled():
    return (
        os.getenv("SPINDLE_BENCHMARK_METRICS") == "1"
        and os.getenv("OTEL_SDK_DISABLED", "").lower() != "true"
    )


def numeric_values(value, prefix=""):
    if isinstance(value, dict):
        for key, item in value.items():
            yield from numeric_values(item, f"{prefix}.{key}" if prefix else str(key))
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            yield from numeric_values(item, f"{prefix}.{index}")
    elif (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    ):
        yield prefix, value


def instrument_name(name):
    normalized = re.sub(r"[^A-Za-z0-9_.\-/]", "_", name)
    if not normalized or not normalized[0].isalpha():
        normalized = "metric." + normalized
    if normalized != name or len(normalized) > 63:
        normalized = (
            normalized[:54] + "." + hashlib.sha256(name.encode()).hexdigest()[:8]
        )
    return normalized


class BenchmarkMetrics:
    def __init__(self, metric_reader=None):
        resource = Resource.create(
            {"service.name": os.getenv("OTEL_SERVICE_NAME", "spindle-benchmarking")}
        )
        self.tags = {
            "spindle.worker": socket.gethostname(),
            **dict(resource.attributes),
        }
        for package in (
            "torch",
            "sglang",
            "miles",
            "megatron-core",
            "modal",
            "tinker",
            "transformers",
        ):
            try:
                self.tags["spindle.runtime." + package] = importlib.metadata.version(
                    package
                )
            except importlib.metadata.PackageNotFoundError:
                continue
        for name, directory in (
            ("miles", "/root/miles"),
            ("megatron", "/root/Megatron-LM"),
        ):
            if os.path.isdir(os.path.join(directory, ".git")):
                try:
                    self.tags["spindle.runtime.commit." + name] = (
                        subprocess.check_output(
                            ["git", "-C", directory, "rev-parse", "HEAD"],
                            text=True,
                            timeout=3,
                        ).strip()
                    )
                except (OSError, subprocess.SubprocessError):
                    continue
        self.provider = MeterProvider(
            resource=resource,
            shutdown_on_exit=False,
            metric_readers=[
                metric_reader
                or PeriodicExportingMetricReader(
                    OTLPMetricExporter(),
                    export_interval_millis=2000,
                    export_timeout_millis=1500,
                )
            ],
        )
        self.meter = self.provider.get_meter("spindle.benchmark")
        self.instruments = {}
        self.lock = threading.Lock()
        self.closed = False
        atexit.register(self.close)

    def close(self):
        with self.lock:
            if self.closed:
                return
            self.closed = True
        self.provider.shutdown(timeout_millis=5000)

    def record(self, metrics, labels=None, prefix=""):
        for name, value in numeric_values(metrics, prefix):
            with self.lock:
                if self.closed:
                    return
                if name not in self.instruments:
                    self.instruments[name] = self.meter.create_gauge(
                        instrument_name(name)
                    )
                instrument = self.instruments[name]
                instrument.set(
                    value,
                    {
                        **self.tags,
                        **(labels or {}),
                        "spindle.metric.original_name": name,
                    },
                )


@functools.cache
def provider():
    if not enabled() or not (
        os.getenv("OTEL_EXPORTER_OTLP_METRICS_ENDPOINT")
        or os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
    ):
        return None
    try:
        return BenchmarkMetrics()
    except Exception:
        logger.warning("Benchmark metrics initialization failed", exc_info=False)
        return None


def record(metrics, labels=None, prefix=""):
    try:
        sink = provider()
        if sink is not None:
            sink.record(metrics, labels, prefix)
    except Exception:
        logger.warning("Benchmark metric recording failed", exc_info=False)


def prometheus_samples(text):
    types = {}
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("# TYPE "):
            _, _, name, kind = line.split(maxsplit=3)
            types[name] = kind
            continue
        if not line or line.startswith("#"):
            continue
        match = SAMPLE.fullmatch(line)
        if not match:
            raise ValueError("unsupported Prometheus sample")
        name, raw_labels, raw_value = match.groups()
        labels = {}
        remaining = raw_labels or ""
        while remaining:
            label = LABEL.match(remaining)
            if label is None:
                raise ValueError("invalid Prometheus label")
            labels[label[1]] = json.loads('"' + label[2] + '"')
            remaining = remaining[label.end() :]
        kind = types.get(name)
        if kind is None:
            kind = next(
                (
                    types[base]
                    for suffix in ("_bucket", "_sum", "_count", "_total", "_created")
                    if name.endswith(suffix)
                    and (base := name.removesuffix(suffix)) in types
                ),
                "untyped",
            )
        yield (
            name,
            float(raw_value),
            {**labels, "prometheus.type": kind, "prometheus.name": name},
        )


GPU_FIELDS = (
    "index",
    "uuid",
    "name",
    "utilization.gpu",
    "utilization.memory",
    "memory.used",
    "memory.total",
    "power.draw",
    "temperature.gpu",
    "clocks.sm",
    "clocks.mem",
)


def scrape_once(urls=(), *, devices=True):
    for url in urls:
        try:
            response = httpx.get(url, timeout=2, trust_env=False)
            response.raise_for_status()
            samples = list(prometheus_samples(response.text))
            for name, value, labels in samples:
                if math.isfinite(value):
                    record({name.replace(":", "."): value}, {**labels, "source": url})
                else:
                    record(
                        {"spindle.collection.nonfinite_samples": 1},
                        {"source": url, "metric": name},
                    )
            record(
                {
                    "spindle.collection.scrape_ok": 1,
                    "spindle.collection.samples": len(samples),
                },
                {"source": url},
            )
        except Exception:
            record({"spindle.collection.scrape_ok": 0}, {"source": url})
            logger.warning("Benchmark Prometheus scrape failed for %s", url)
    if devices:
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=" + ",".join(GPU_FIELDS),
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                check=True,
                timeout=3,
            )
            for row in result.stdout.splitlines():
                fields = dict(
                    zip(GPU_FIELDS, (x.strip() for x in row.split(",")), strict=True)
                )
                values = {}
                for name in GPU_FIELDS[3:]:
                    try:
                        values[name] = float(fields[name])
                    except ValueError:
                        continue
                record(
                    values,
                    {
                        "gpu.index": fields["index"],
                        "gpu.uuid": fields["uuid"],
                        "gpu.name": fields["name"],
                    },
                    "spindle.gpu",
                )
            record({"spindle.collection.device_ok": 1})
        except Exception:
            record({"spindle.collection.device_ok": 0})


def start_collection(*, urls=(), process=None, observe=None, devices=True):
    stop = threading.Event()
    if not enabled() or provider() is None:
        return stop
    configured = json.loads(os.getenv("SPINDLE_BENCHMARK_PROMETHEUS_URLS", "[]"))
    urls = tuple(dict.fromkeys([*urls, *configured]))

    def collect():
        while not stop.is_set() and (process is None or process.poll() is None):
            scrape_once(urls, devices=devices)
            if observe is not None:
                try:
                    observe()
                except Exception:
                    record({"spindle.collection.rank_ok": 0})
            stop.wait(2)

    thread = threading.Thread(
        target=collect, name="spindle-benchmark-metrics", daemon=True
    )
    thread.start()
    atexit.register(stop.set)
    return stop


def start_cuda_collection(torch):
    if not enabled():
        return
    device = torch.cuda.current_device()
    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0

    def observe():
        record(
            dict(torch.cuda.memory_stats(device)),
            {"rank": rank, "device": device},
            "spindle.cuda.memory",
        )
        free, total = torch.cuda.mem_get_info(device)
        record(
            {"free_bytes": free, "total_bytes": total},
            {"rank": rank, "device": device},
            "spindle.cuda.device",
        )

    return start_collection(observe=observe, devices=False)
