"""
FastAPI application — gateway for the RAG Ingestion Engine.

Endpoints:
  - GET  /health          → liveness probe
  - GET  /health/ready    → readiness probe (DB + Redis + Celery)
  - GET  /metrics         → Prometheus metrics
  - POST /api/ingest      → async document ingestion (returns 202)
  - GET  /api/status/{id} → Celery job status & progress
  - POST /api/search      → semantic search with Redis cache + pgvector ANN
"""

import logging
import time
from contextlib import asynccontextmanager

import redis
from celery.result import AsyncResult
from fastapi import FastAPI, HTTPException, Request, Response
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
from sqlalchemy import text

from app.celery_app import celery_app
from app.config import settings
from app.database import SessionLocal, engine, init_db
from app.embeddings import encode_single
from app.logging_config import setup_logging
from app.metrics import (
    CACHE_HITS,
    CACHE_MISSES,
    HTTP_REQUEST_DURATION,
    HTTP_REQUESTS_TOTAL,
)
from app.models import DocumentChunk
from app.schemas import (
    ChunkResult,
    IngestRequest,
    IngestResponse,
    JobStatus,
    SearchRequest,
    SearchResponse,
)
from app.semantic_cache import SemanticCache

logger = logging.getLogger(__name__)


# ── Lifespan: bootstrap DB tables & indexes on startup ─────────────────
@asynccontextmanager
async def lifespan(application: FastAPI):
    """Run one-time startup work, then yield to serve requests."""
    setup_logging()
    logger.info("Starting RAG Ingestion Engine...")
    init_db()
    application.state.semantic_cache = SemanticCache()
    logger.info("SemanticCache initialised.")
    yield
    logger.info("Shutting down RAG Ingestion Engine.")


app = FastAPI(
    title="RAG Ingestion Engine",
    description=(
        "Distributed high-throughput ingestion pipeline with semantic caching. "
        "Accepts raw text documents, asynchronously chunks & embeds them via "
        "Celery workers, and stores vectors in PostgreSQL/pgvector with HNSW "
        "indexing. Features Redis-backed semantic caching and token-bucket "
        "rate limiting."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)


# ═══════════════════════════════════════════════════════════════════════
# Middleware — Prometheus HTTP metrics
# ═══════════════════════════════════════════════════════════════════════
@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    """Track HTTP request latency and count for every endpoint."""
    start = time.perf_counter()
    response = await call_next(request)
    duration = time.perf_counter() - start

    path = request.url.path
    method = request.method
    status = str(response.status_code)

    HTTP_REQUEST_DURATION.labels(method=method, path=path, status_code=status).observe(duration)
    HTTP_REQUESTS_TOTAL.labels(method=method, path=path, status_code=status).inc()

    return response


# ═══════════════════════════════════════════════════════════════════════
# Health Endpoints
# ═══════════════════════════════════════════════════════════════════════
@app.get("/health", tags=["health"])
def health():
    """Liveness probe — always returns 200 if the process is running."""
    return {"status": "healthy"}


@app.get("/health/ready", tags=["health"])
def readiness():
    """
    Readiness probe — verifies connectivity to PostgreSQL, Redis,
    and reports whether at least one Celery worker is reachable.
    """
    checks: dict = {}

    # ── PostgreSQL ──
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        checks["db"] = "ok"
    except Exception as exc:
        checks["db"] = f"error: {exc}"

    # ── Redis ──
    try:
        r = redis.from_url(settings.REDIS_URL)
        r.ping()
        checks["redis"] = "ok"
    except Exception as exc:
        checks["redis"] = f"error: {exc}"

    # ── Celery ──
    try:
        inspector = celery_app.control.inspect(timeout=2.0)
        ping_result = inspector.ping()
        checks["celery"] = "ok" if ping_result else "no workers"
    except Exception as exc:
        checks["celery"] = f"error: {exc}"

    all_ok = all(v == "ok" for v in checks.values())
    return {
        "status": "ready" if all_ok else "degraded",
        "services": checks,
    }


# ═══════════════════════════════════════════════════════════════════════
# Prometheus Metrics Endpoint
# ═══════════════════════════════════════════════════════════════════════
@app.get("/metrics", tags=["observability"], include_in_schema=False)
def prometheus_metrics():
    """Expose Prometheus-compatible metrics."""
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST,
    )


# ═══════════════════════════════════════════════════════════════════════
# Ingestion Endpoints
# ═══════════════════════════════════════════════════════════════════════
@app.post(
    "/api/ingest",
    response_model=IngestResponse,
    status_code=202,
    tags=["ingestion"],
    summary="Ingest a text document",
    description="Accepts a raw text document for asynchronous ingestion. "
                "Returns 202 Accepted immediately with a job_id.",
)
def ingest(request: IngestRequest):
    """
    Accept a raw text document for asynchronous ingestion.

    Returns **202 Accepted** immediately with a ``job_id`` that can be
    polled via ``GET /api/status/{job_id}``.
    """
    from app.tasks import ingest_document

    task = ingest_document.delay(request.text, request.metadata)
    logger.info("Ingestion job enqueued.", extra={"task_id": task.id})
    return IngestResponse(job_id=task.id)


@app.get(
    "/api/status/{job_id}",
    response_model=JobStatus,
    tags=["ingestion"],
    summary="Check ingestion job status",
)
def job_status(job_id: str):
    """
    Check the progress of an ingestion job via the Celery result backend.

    Possible states: PENDING → STARTED → PROGRESS → SUCCESS | FAILURE.
    """
    result = AsyncResult(job_id, app=celery_app)

    response = JobStatus(job_id=job_id, status=result.state)

    if result.state == "PROGRESS":
        response.progress = result.info or {}
    elif result.state == "SUCCESS":
        response.result = result.result
    elif result.state == "FAILURE":
        response.error = str(result.result)

    return response


# ═══════════════════════════════════════════════════════════════════════
# Search Endpoint
# ═══════════════════════════════════════════════════════════════════════
@app.post(
    "/api/search",
    response_model=SearchResponse,
    tags=["search"],
    summary="Semantic vector search",
    description="Two-tier search: checks Redis semantic cache first "
                "(cosine ≥ 0.95), falls back to pgvector ANN search.",
)
def search(request: SearchRequest):
    """
    Semantic vector search with a two-tier strategy:

    1. **Semantic Cache** (Redis): If a cached query has cosine
       similarity ≥ 0.95, return instantly — bypasses pgvector entirely.
    2. **pgvector ANN** (PostgreSQL HNSW): Falls back to approximate
       nearest-neighbor search, caches the result for future hits.
    """
    cache: SemanticCache = app.state.semantic_cache

    # ── 1. Generate query embedding ──
    query_embedding = encode_single(request.query)

    # ── 2. Check semantic cache ──
    cached = cache.get(query_embedding)
    if cached is not None:
        CACHE_HITS.inc()
        logger.info("Semantic cache HIT for query: %s", request.query[:80])
        chunk_items = [ChunkResult(**c) if isinstance(c, dict) else c for c in cached] if isinstance(cached, list) else []
        return SearchResponse(
            query=request.query,
            cached=True,
            results=chunk_items,
        )

    CACHE_MISSES.inc()

    # ── 3. Fallback: pgvector ANN search ──
    db = SessionLocal()
    try:
        # Use pgvector's <=> operator for cosine distance
        results = (
            db.query(
                DocumentChunk,
                DocumentChunk.embedding.cosine_distance(query_embedding).label(
                    "distance"
                ),
            )
            .order_by("distance")
            .limit(request.top_k)
            .all()
        )

        chunks = []
        for chunk, distance in results:
            score = 1.0 - distance  # cosine similarity
            chunks.append(
                ChunkResult(
                    id=str(chunk.id),
                    document_id=chunk.document_id,
                    chunk_index=chunk.chunk_index,
                    content=chunk.content,
                    score=round(score, 4),
                    metadata=chunk.metadata_ or {},
                )
            )
    finally:
        db.close()

    # ── 4. Populate semantic cache for future queries ──
    if chunks:
        cache.set(
            query=request.query,
            embedding=query_embedding,
            result=[c.model_dump() for c in chunks],
        )
        logger.info("Cached %d results for query: %s", len(chunks), request.query[:80])

    return SearchResponse(
        query=request.query,
        cached=False,
        results=chunks,
    )
