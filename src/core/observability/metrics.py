"""
OpenTelemetry Metrics

Phase 15: Observability

Provides application metrics collection.
"""

import logging
from typing import Optional, Dict, Any

from opentelemetry import metrics
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import (
    ConsoleMetricExporter,
    PeriodicExportingMetricReader,
)
from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
from opentelemetry.sdk.resources import Resource, SERVICE_NAME

logger = logging.getLogger(__name__)

# Global meter
_meter: Optional[metrics.Meter] = None

# Metric instruments
_counters: Dict[str, metrics.Counter] = {}
_histograms: Dict[str, metrics.Histogram] = {}


def init_metrics(
    service_name: str = "zakops-backend",
    otlp_endpoint: Optional[str] = None,
    console_export: bool = False,
    export_interval_ms: int = 60000
) -> metrics.Meter:
    """
    Initialize OpenTelemetry metrics.

    Args:
        service_name: Name of the service
        otlp_endpoint: OTLP exporter endpoint
        console_export: Enable console export for debugging
        export_interval_ms: Export interval in milliseconds

    Returns:
        Configured meter
    """
    global _meter

    readers = []

    if otlp_endpoint:
        otlp_exporter = OTLPMetricExporter(endpoint=otlp_endpoint, insecure=True)
        readers.append(PeriodicExportingMetricReader(
            otlp_exporter,
            export_interval_millis=export_interval_ms
        ))
        logger.info(f"OTel metrics: OTLP exporter configured -> {otlp_endpoint}")

    if console_export:
        readers.append(PeriodicExportingMetricReader(
            ConsoleMetricExporter(),
            export_interval_millis=export_interval_ms
        ))
        logger.info("OTel metrics: Console exporter enabled")

    resource = Resource.create({SERVICE_NAME: service_name})

    provider = MeterProvider(resource=resource, metric_readers=readers)
    metrics.set_meter_provider(provider)

    _meter = metrics.get_meter(service_name)

    # Initialize standard metrics
    _init_standard_metrics()

    logger.info(f"OTel metrics initialized: {service_name}")

    return _meter


def _init_standard_metrics():
    """Initialize standard application metrics."""
    global _counters, _histograms

    meter = get_meter()

    # Request counters
    _counters["http_requests_total"] = meter.create_counter(
        "http_requests_total",
        description="Total HTTP requests",
        unit="1"
    )

    _counters["agent_invocations_total"] = meter.create_counter(
        "agent_invocations_total",
        description="Total agent invocations",
        unit="1"
    )

    _counters["actions_created_total"] = meter.create_counter(
        "actions_created_total",
        description="Total actions created",
        unit="1"
    )

    _counters["events_published_total"] = meter.create_counter(
        "events_published_total",
        description="Total events published",
        unit="1"
    )

    _counters["dlq_entries_total"] = meter.create_counter(
        "dlq_entries_total",
        description="Total DLQ entries",
        unit="1"
    )

    _counters["outbox_processed_total"] = meter.create_counter(
        "outbox_processed_total",
        description="Total outbox entries processed",
        unit="1"
    )

    _counters["sse_connections_total"] = meter.create_counter(
        "sse_connections_total",
        description="Total SSE connections opened",
        unit="1"
    )

    # Histograms
    _histograms["http_request_duration_seconds"] = meter.create_histogram(
        "http_request_duration_seconds",
        description="HTTP request duration",
        unit="s"
    )

    _histograms["agent_run_duration_seconds"] = meter.create_histogram(
        "agent_run_duration_seconds",
        description="Agent run duration",
        unit="s"
    )

    _histograms["outbox_processing_duration_seconds"] = meter.create_histogram(
        "outbox_processing_duration_seconds",
        description="Outbox processing duration",
        unit="s"
    )

    _histograms["db_query_duration_seconds"] = meter.create_histogram(
        "db_query_duration_seconds",
        description="Database query duration",
        unit="s"
    )


def get_meter() -> metrics.Meter:
    """Get the global meter."""
    global _meter
    if _meter is None:
        _meter = metrics.get_meter("zakops-backend")
    return _meter


def record_counter(
    name: str,
    value: int = 1,
    attributes: Dict[str, Any] = None
):
    """Record a counter metric."""
    if name in _counters:
        _counters[name].add(value, attributes or {})


def record_histogram(
    name: str,
    value: float,
    attributes: Dict[str, Any] = None
):
    """Record a histogram metric."""
    if name in _histograms:
        _histograms[name].record(value, attributes or {})
