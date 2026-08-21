"""
Celery tasks — asynchronous document ingestion pipeline.

Flow:
  1. Receive raw text + metadata.
  2. Chunk text into ~512-token windows with overlap.
  3. Embed chunks in dynamic micro-batches (default 32) gated by TokenBucket.
  4. Bulk-insert embeddings into PostgreSQL / pgvector.

Resilience:
  - Automatic retries with exponential backoff on transient failures.
  - Prometheus metrics for throughput, errors, and latency.
  - Structured JSON logging with task_id context.
"""

import logging
import time
import uuid

from celery import Task
from sqlalchemy.exc import OperationalError

from app.celery_app import celery_app
from app.config import settings
from app.database import SessionLocal
from app.metrics import (
    ACTIVE_INGESTION_TASKS,
    CHUNKS_PROCESSED,
    DOCUMENTS_INGESTED,
    INGESTION_DURATION,
    INGESTION_ERRORS,
)
from app.models import DocumentChunk

logger = logging.getLogger(__name__)


# ── Text Chunker ───────────────────────────────────────────────────────
def chunk_text(
    text: str,
    chunk_size: int = settings.CHUNK_SIZE,
    overlap: int = settings.CHUNK_OVERLAP,
) -> list[str]:
    """
    Split *text* into chunks of approximately *chunk_size* whitespace
    tokens, with *overlap* tokens shared between consecutive chunks.

    Returns an empty list for empty/whitespace-only input.
    """
    words = text.split()
    if not words:
        return []

    chunks: list[str] = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        if end >= len(words):
            break
        start += chunk_size - overlap
    return chunks


# ── Ingestion Task ─────────────────────────────────────────────────────
@celery_app.task(
    bind=True,
    name="ingest_document",
    # ── Retry policy: exponential backoff on transient failures ──
    autoretry_for=(OperationalError, ConnectionError, OSError),
    retry_backoff=True,          # exponential: 1s, 2s, 4s, 8s, ...
    retry_backoff_max=300,       # cap at 5 minutes
    retry_jitter=True,           # add randomness to prevent thundering herd
    max_retries=5,
    # ── Ack late so failed tasks are re-delivered ──
    acks_late=True,
)
def ingest_document(self: Task, text: str, metadata: dict | None = None) -> dict:
    """
    Full ingestion pipeline:
      chunk → embed (with dynamic batching & rate limiting) → store.

    Retries automatically on DB connection errors and network failures
    with exponential backoff (up to 5 retries).
    """
    from app.embeddings import encode  # deferred import for worker process
    from app.rate_limiter import TokenBucket

    metadata = metadata or {}
    document_id = uuid.uuid4().hex
    start_time = time.perf_counter()

    ACTIVE_INGESTION_TASKS.inc()

    try:
        return _run_ingestion(self, text, metadata, document_id, encode, TokenBucket)
    except (OperationalError, ConnectionError, OSError):
        # Let Celery's autoretry handle these
        INGESTION_ERRORS.inc()
        raise
    except Exception as exc:
        INGESTION_ERRORS.inc()
        logger.exception(
            "Ingestion failed for document %s: %s",
            document_id,
            exc,
            extra={"task_id": self.request.id},
        )
        raise
    finally:
        duration = time.perf_counter() - start_time
        INGESTION_DURATION.observe(duration)
        ACTIVE_INGESTION_TASKS.dec()


def _run_ingestion(
    task: Task,
    text: str,
    metadata: dict,
    document_id: str,
    encode,
    TokenBucket,
) -> dict:
    """Core ingestion logic, separated for testability."""

    # ── 1. Chunk ──
    chunks = chunk_text(text)
    total_chunks = len(chunks)

    if total_chunks == 0:
        logger.warning("Document %s has no content to ingest.", document_id)
        return {
            "document_id": document_id,
            "total_chunks": 0,
            "status": "completed",
        }

    logger.info(
        "Document %s: %d chunks (chunk_size=%d, overlap=%d).",
        document_id,
        total_chunks,
        settings.CHUNK_SIZE,
        settings.CHUNK_OVERLAP,
        extra={"task_id": task.request.id},
    )

    task.update_state(
        state="PROGRESS",
        meta={
            "document_id": document_id,
            "total_chunks": total_chunks,
            "processed": 0,
        },
    )

    # ── 2. Dynamic micro-batching with rate limiting ──
    batch_size = settings.BATCH_SIZE
    bucket = TokenBucket(
        capacity=settings.RATE_LIMIT_CAPACITY,
        refill_rate=settings.RATE_LIMIT_REFILL_RATE,
    )

    processed = 0
    db = SessionLocal()
    try:
        for batch_start in range(0, total_chunks, batch_size):
            batch_end = min(batch_start + batch_size, total_chunks)
            batch_texts = chunks[batch_start:batch_end]
            batch_len = len(batch_texts)

            # Rate-limit: block until enough tokens are available
            bucket.acquire(tokens=batch_len, blocking=True)

            # ── 3. Embed batch ──
            embeddings = encode(batch_texts)

            # ── 4. Bulk-insert ──
            chunk_objects = []
            for i, (chunk_text_item, emb) in enumerate(
                zip(batch_texts, embeddings), start=batch_start
            ):
                chunk_objects.append(
                    DocumentChunk(
                        document_id=document_id,
                        chunk_index=i,
                        content=chunk_text_item,
                        embedding=emb.tolist(),
                        metadata_=metadata,
                    )
                )

            db.bulk_save_objects(chunk_objects)
            db.commit()

            processed += batch_len
            CHUNKS_PROCESSED.inc(batch_len)

            task.update_state(
                state="PROGRESS",
                meta={
                    "document_id": document_id,
                    "total_chunks": total_chunks,
                    "processed": processed,
                },
            )
            logger.info(
                "Document %s: batch %d–%d embedded & stored (%d/%d).",
                document_id,
                batch_start,
                batch_end - 1,
                processed,
                total_chunks,
                extra={"task_id": task.request.id},
            )
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()

    DOCUMENTS_INGESTED.inc()

    result = {
        "document_id": document_id,
        "total_chunks": total_chunks,
        "status": "completed",
    }
    logger.info(
        "Document %s ingestion complete.",
        document_id,
        extra={"task_id": task.request.id},
    )
    return result
