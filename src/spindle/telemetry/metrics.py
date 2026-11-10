"""Private OTLP meter providers shared by trainer instrumentation."""

import os

from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.sdk.resources import Resource


def meter_provider(metric_reader=None):
    """Return a private provider and its resource, or (None, None) when disabled."""
    if metric_reader is None and (
        os.getenv("OTEL_SDK_DISABLED", "").lower() == "true"
        or not (
            os.getenv("OTEL_EXPORTER_OTLP_METRICS_ENDPOINT")
            or os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
        )
    ):
        return None, None
    protocol = os.getenv(
        "OTEL_EXPORTER_OTLP_METRICS_PROTOCOL",
        os.getenv("OTEL_EXPORTER_OTLP_PROTOCOL", "http/protobuf"),
    )
    if protocol != "http/protobuf":
        raise ValueError("Trainer metrics require http/protobuf")
    reader = metric_reader or PeriodicExportingMetricReader(
        OTLPMetricExporter(),
        export_interval_millis=5000,
        export_timeout_millis=4000,
    )
    resource = Resource.create(
        {"service.name": os.getenv("OTEL_SERVICE_NAME", "spindle")}
    )
    return MeterProvider(resource=resource, metric_readers=[reader]), resource
