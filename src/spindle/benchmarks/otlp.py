from __future__ import annotations

import base64
import time

from google.protobuf.json_format import MessageToDict
from opentelemetry.proto.collector.metrics.v1.metrics_service_pb2 import (
    ExportMetricsServiceRequest,
)
from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import (
    ExportTraceServiceRequest,
)


def attributes(items):
    def value(v):
        field = v.WhichOneof("value")
        if field == "array_value":
            return [value(x) for x in v.array_value.values]
        if field == "kvlist_value":
            return attributes(v.kvlist_value.values)
        if field == "bytes_value":
            return base64.b64encode(v.bytes_value).decode()
        return getattr(v, field) if field else None

    return {item.key: value(item.value) for item in items}


def decode_metrics(data):
    request = ExportMetricsServiceRequest.FromString(data)
    for resource in request.resource_metrics:
        resource_attrs = attributes(resource.resource.attributes)
        if not resource_attrs.get("spindle.run_id") and not resource.scope_metrics:
            raise ValueError("OTLP resources require spindle.run_id")
        for scope in resource.scope_metrics:
            for metric in scope.metrics:
                metric_type = metric.WhichOneof("data")
                if not metric_type:
                    continue
                for point in getattr(metric, metric_type).data_points:
                    point_labels = attributes(point.attributes)
                    run_id = point_labels.get("spindle.run_id") or resource_attrs.get(
                        "spindle.run_id"
                    )
                    if not run_id:
                        raise ValueError("OTLP points require spindle.run_id")
                    name = (
                        point_labels.get("prometheus.name")
                        or point_labels.get("spindle.metric.original_name")
                        or metric.name
                    )
                    raw = MessageToDict(point, preserving_proto_field_name=True)
                    values = {}
                    if metric_type in ("gauge", "sum"):
                        values[name] = (
                            point.as_int
                            if point.WhichOneof("value") == "as_int"
                            else point.as_double
                        )
                    else:
                        values[name] = {
                            k: getattr(point, k) for k in ("count", "sum") if k in raw
                        }
                        if "min" in raw:
                            values[name]["min"] = point.min
                        if "max" in raw:
                            values[name]["max"] = point.max
                        if metric_type == "histogram":
                            values[name]["bucket_counts"] = list(point.bucket_counts)
                            values[name]["explicit_bounds"] = list(
                                point.explicit_bounds
                            )
                        elif metric_type == "exponential_histogram":
                            values[name].update(
                                zero_count=point.zero_count,
                                scale=point.scale,
                                positive_buckets=list(point.positive.bucket_counts),
                                negative_buckets=list(point.negative.bucket_counts),
                            )
                        elif metric_type == "summary":
                            values[name]["quantiles"] = {
                                str(q.quantile): q.value for q in point.quantile_values
                            }
                    yield (
                        str(run_id),
                        {
                            "kind": "otlp." + metric_type,
                            "time": point.time_unix_nano / 1e9 or time.time(),
                            "labels": {
                                **resource_attrs,
                                **attributes(point.attributes),
                                "otel.scope": scope.scope.name,
                            },
                            "unit": metric.unit,
                            "metrics": values,
                            "raw": {
                                "point": raw,
                                "name": metric.name,
                                "description": metric.description,
                                "temporality": getattr(
                                    getattr(metric, metric_type),
                                    "aggregation_temporality",
                                    None,
                                ),
                            },
                        },
                    )


def decode_traces(data):
    request = ExportTraceServiceRequest.FromString(data)
    for resource in request.resource_spans:
        resource_attrs = attributes(resource.resource.attributes)
        if not resource_attrs.get("spindle.run_id") and not resource.scope_spans:
            raise ValueError("OTLP resources require spindle.run_id")
        for scope in resource.scope_spans:
            for span in scope.spans:
                span_attributes = attributes(span.attributes)
                run_id = span_attributes.get("spindle.run_id") or resource_attrs.get(
                    "spindle.run_id"
                )
                if not run_id:
                    raise ValueError("OTLP spans require spindle.run_id")
                yield (
                    str(run_id),
                    {
                        "kind": "span",
                        "time": span.start_time_unix_nano / 1e9,
                        "labels": {
                            **resource_attrs,
                            **span_attributes,
                            "span.name": span.name,
                        },
                        "metrics": {
                            "span.duration_s": (
                                span.end_time_unix_nano - span.start_time_unix_nano
                            )
                            / 1e9,
                            **{
                                k: v
                                for k, v in span_attributes.items()
                                if isinstance(v, (int, float))
                                and not isinstance(v, bool)
                            },
                        },
                        "raw": MessageToDict(span, preserving_proto_field_name=True),
                    },
                )
