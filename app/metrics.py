"""
Prometheus metrics for observability.

Tracks:
  - HTTP request latency & count (by method, path, status)
  - Ingestion throughput (documents ingested, chunks processed)
  - Semantic cache hit/miss ratio
  - Embedding generation latency
"""

import time
from functools import wraps

from prometheus_client import Counter, Histogram, Gauge, Info


# ── Application Info ───────────────────────────────────────────────────
APP_INFO = Info("rag_engine", "RAG Ingestion Engine metadata")
APP_INFO.info({"version": "1.0.0", "embedding_model": "all-MiniLM-L6-v2"})

# ── HTTP Metrics ───────────────────────────────────────────────────────
HTTP_REQUEST_DURATION = Histogram(
    "http_request_duration_seconds",
    "HTTP request latency in seconds",
    labelnames=["method", "path", "status_code"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
)

HTTP_REQUESTS_TOTAL = Counter(
    "http_requests_total",
    "Total HTTP requests",
    labelnames=["method", "path", "status_code"],
)

# ── Ingestion Metrics ─────────────────────────────────────────────────
DOCUMENTS_INGESTED = Counter(
    "documents_ingested_total",
    "Total documents successfully ingested",
)

CHUNKS_PROCESSED = Counter(
    "chunks_processed_total",
    "Total chunks embedded and stored",
)

INGESTION_DURATION = Histogram(
    "ingestion_duration_seconds",
    "End-to-end ingestion task duration",
    buckets=(0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0),
)

INGESTION_ERRORS = Counter(
    "ingestion_errors_total",
    "Total ingestion task failures",
)

# ── Semantic Cache Metrics ────────────────────────────────────────────
CACHE_HITS = Counter(
    "semantic_cache_hits_total",
    "Total semantic cache hits",
)

CACHE_MISSES = Counter(
    "semantic_cache_misses_total",
    "Total semantic cache misses",
)

# ── Embedding Metrics ────────────────────────────────────────────────
EMBEDDING_DURATION = Histogram(
    "embedding_duration_seconds",
    "Time to generate embeddings for a batch",
    buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5),
)

EMBEDDING_BATCH_SIZE = Histogram(
    "embedding_batch_size",
    "Number of texts per embedding batch",
    buckets=(1, 4, 8, 16, 32, 64, 128),
)

# ── Active Workers ───────────────────────────────────────────────────
ACTIVE_INGESTION_TASKS = Gauge(
    "active_ingestion_tasks",
    "Currently running ingestion tasks",
)


def track_embedding_time(func):
    """Decorator to measure embedding generation latency."""
    @wraps(func)
    def wrapper(texts, *args, **kwargs):
        EMBEDDING_BATCH_SIZE.observe(len(texts))
        start = time.perf_counter()
        result = func(texts, *args, **kwargs)
        EMBEDDING_DURATION.observe(time.perf_counter() - start)
        return result
    return wrapper
