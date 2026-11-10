from datetime import datetime, timedelta
from types import SimpleNamespace
import time

import pytest
import torch.monitor as monitor
from opentelemetry.sdk.metrics.export import InMemoryMetricReader

from spindle.telemetry import metrics, torch_monitor


@pytest.fixture
def bridge(monkeypatch):
    reader = InMemoryMetricReader()
    provider, resource = metrics.meter_provider(reader)
    monkeypatch.setattr(torch_monitor, "meter_provider", lambda: (provider, resource))
    monkeypatch.setattr(torch_monitor, "_handler", None)
    monkeypatch.setattr(torch_monitor, "_provider", None)
    monkeypatch.setattr(torch_monitor, "_warned", False)
    monkeypatch.setenv("SPINDLE_TORCH_MONITOR_METRICS", "true")
    monkeypatch.setenv("SPINDLE_DEFINITION_ID", "definition")
    monkeypatch.setenv("RANK", "3")
    yield reader
    if torch_monitor._handler is not None:
        monitor.unregister_event_handler(torch_monitor._handler)
    provider.shutdown()


def points(reader):
    data = reader.get_metrics_data()
    if data is None:
        return {}
    return {
        metric.name: metric.data.data_points
        for resource in data.resource_metrics
        for scope in resource.scope_metrics
        for metric in scope.metrics
    }


def test_events_and_stat_windows(bridge):
    torch_monitor.install(component="trainer-rank")
    handler = torch_monitor._handler
    torch_monitor.install()
    assert torch_monitor._handler is handler
    monitor.log_event(monitor.Event("custom", datetime.now(), {"loss/value": 2.5}))
    stat = monitor.Stat(
        "window",
        [monitor.Aggregation.SUM, monitor.Aggregation.COUNT],
        timedelta(milliseconds=100),
    )
    stat.add(7)
    time.sleep(0.15)
    stat.add(9)
    data = points(bridge)
    for name, value, event in (
        ("loss_value", 2.5, "custom"),
        ("window.sum", 7, "torch.monitor.Stat"),
        ("window.count", 1, "torch.monitor.Stat"),
    ):
        (point,) = data["torch.monitor." + name]
        assert point.value == value
        assert dict(point.attributes) == {
            "torch.monitor.event": event,
            "spindle.rank": 3,
            "spindle.definition_id": "definition",
            "spindle.component": "trainer-rank",
        }
    del stat
    assert points(bridge)["torch.monitor.window.sum"][0].value == 9


@pytest.mark.parametrize("disabled", ["flag", "endpoint", "sdk"])
def test_disabled_registers_nothing(bridge, monkeypatch, disabled):
    if disabled == "flag":
        monkeypatch.setenv("SPINDLE_TORCH_MONITOR_METRICS", "0")
    else:
        monkeypatch.setattr(torch_monitor, "meter_provider", metrics.meter_provider)
        monkeypatch.delenv("OTEL_EXPORTER_OTLP_ENDPOINT", raising=False)
        monkeypatch.delenv("OTEL_EXPORTER_OTLP_METRICS_ENDPOINT", raising=False)
        if disabled == "sdk":
            monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4318")
            monkeypatch.setenv("OTEL_SDK_DISABLED", "true")
    torch_monitor.install()
    assert torch_monitor._handler is None
    monitor.log_event(monitor.Event("disabled", datetime.now(), {"value": 1}))
    assert points(bridge) == {}


def test_filter_cap_and_distributed_rank(bridge, monkeypatch):
    import torch.distributed as dist

    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "get_rank", lambda: 8)
    monkeypatch.setattr(torch_monitor, "MAX_INSTRUMENTS", 2)
    torch_monitor.install()
    monitor.log_event(monitor.Event("empty", datetime.now(), {}))
    monitor.log_event(
        monitor.Event(
            "mixed",
            datetime.now(),
            {
                "string": "no",
                "nan": float("nan"),
                "inf": float("inf"),
                "first": 1,
                "second": 2,
                "third": 3,
            },
        )
    )
    data = points(bridge)
    assert len(data) == 2
    expected = {
        "torch.monitor.first": 1,
        "torch.monitor.second": 2,
        "torch.monitor.third": 3,
    }
    for name, (point,) in data.items():
        assert point.value == expected[name]
        assert point.attributes["spindle.rank"] == 8


def test_handler_errors_warn_once_and_recover(bridge, monkeypatch, caplog):
    captured = []
    register = monitor.register_event_handler

    def capture(handler):
        captured.append(handler)
        return register(handler)

    monkeypatch.setattr(monitor, "register_event_handler", capture)
    torch_monitor.install()
    for _ in range(2):
        captured[0](SimpleNamespace(name="broken", data=None))
    captured[0](SimpleNamespace(name="bool", data={"bool": True}))
    monitor.log_event(monitor.Event("valid", datetime.now(), {"value": 4}))
    assert points(bridge)["torch.monitor.value"][0].value == 4
    assert caplog.text.count("Torch monitor metrics failed") == 1


def test_initialization_failure_is_optional(monkeypatch, caplog):
    monkeypatch.setattr(torch_monitor, "_handler", None)
    monkeypatch.setattr(torch_monitor, "_warned", False)
    monkeypatch.setenv("SPINDLE_TORCH_MONITOR_METRICS", "true")
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4318")
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_METRICS_PROTOCOL", "grpc")
    torch_monitor.install()
    assert torch_monitor._handler is None
    assert "Torch monitor metrics failed" in caplog.text
