"""Prometheus metrics for the router and the transformer.

Exposes a small set of histograms / counters under /metrics. KServe
already emits its own metrics for the predictors so we focus on the
router-level path: how long the preprocess step takes, queue depth,
fallback events.
"""
from __future__ import annotations

from prometheus_client import (
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)
from prometheus_client.exposition import CONTENT_TYPE_LATEST

REGISTRY = CollectorRegistry()

REQUEST_LATENCY = Histogram(
    "msa_request_latency_seconds",
    "End-to-end latency from the router perspective.",
    labelnames=("model", "phase", "status"),
    buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30),
    registry=REGISTRY,
)

REQUESTS_TOTAL = Counter(
    "msa_requests_total",
    "Total requests handled by the router.",
    labelnames=("model", "status"),
    registry=REGISTRY,
)

FALLBACK_TOTAL = Counter(
    "msa_fallback_total",
    "How often the router fell back to a secondary model.",
    labelnames=("primary", "fallback"),
    registry=REGISTRY,
)

QUEUE_DEPTH = Gauge(
    "msa_queue_depth",
    "Approx in-flight requests per model (router-side).",
    labelnames=("model",),
    registry=REGISTRY,
)

COLD_STARTS = Counter(
    "msa_cold_starts_total",
    "Times a request hit a freshly scaled-from-zero pod.",
    labelnames=("model",),
    registry=REGISTRY,
)


def render() -> tuple[bytes, str]:
    return generate_latest(REGISTRY), CONTENT_TYPE_LATEST
