# System Design Document

## 1. Problem Statement

Build a backend system that ingests raw text documents, converts them into semantically searchable vector embeddings, and serves low-latency similarity search — all at scale, with caching and rate limiting.

This is the **Retrieval** layer of a RAG (Retrieval-Augmented Generation) system. It does not generate answers — it finds the most relevant text chunks to feed into an LLM.

---

## 2. Requirements

### Functional
- Accept text documents via API and process them asynchronously.
- Chunk documents into overlapping windows, embed each chunk, and store vectors.
- Search stored vectors by semantic similarity (ANN).
- Cache semantically similar queries to avoid redundant computation.
- Track ingestion job progress.

### Non-Functional
- **Non-blocking ingestion**: The API must never block on document processing.
- **Horizontal scalability**: More Celery workers = more throughput, with no code changes.
- **Sub-100ms search latency** on cache hits.
- **Observability**: Prometheus metrics, structured JSON logging.
- **Resilience**: Automatic retries with exponential backoff on transient failures.

---

## 3. Architecture

```
┌─────────────┐         ┌─────────────────────────────────┐
│   Client    │────────▶│       FastAPI Gateway            │
│             │◀────────│  (api service, port 8000)        │
└─────────────┘         │                                  │
                        │  /api/ingest  → enqueue task     │
                        │  /api/search  → cache → pgvector │
                        │  /api/status  → poll Celery      │
                        │  /metrics     → Prometheus       │
                        └──────┬───────────────┬───────────┘
                               │               │
                    ┌──────────▼──┐    ┌───────▼────────┐
                    │    Redis    │    │   PostgreSQL   │
                    │  (Stack)    │    │  + pgvector    │
                    │             │    │                │
                    │ • Broker    │    │ • HNSW index   │
                    │ • Backend   │    │ • 384-dim vecs │
                    │ • Sem.Cache │    │ • JSONB meta   │
                    └──────┬──────┘    └────────────────┘
                           │
                    ┌──────▼──────┐
                    │   Celery    │
                    │   Worker    │
                    │             │
                    │ • Chunking  │
                    │ • Embedding │
                    │ • Batching  │
                    │ • Rate Limit│
                    └─────────────┘
```

---

## 4. Key Design Decisions

### 4.1 Why HNSW over IVFFlat?

| Factor | HNSW | IVFFlat |
|--------|------|---------|
| **Query latency** | O(log n) — consistently fast | Varies by cluster distribution |
| **Build time** | Slower initial build | Faster build |
| **Recall at 95%+** | Excellent | Requires more probes |
| **Update cost** | Inserts without rebuild | Requires periodic re-training |
| **Our use case** | Continuous ingestion (inserts happen all the time) | Better for static datasets |

**Decision**: HNSW. Our system continuously ingests documents, so we need an index that handles inserts without rebuilding. IVFFlat would require periodic `REINDEX` operations as the data distribution shifts, adding operational complexity.

HNSW parameters: `m=16` (connections per node), `ef_construction=200` (build-time accuracy). These provide high recall (>95%) with reasonable memory overhead.

### 4.2 Why Token Bucket over Leaky Bucket?

| Factor | Token Bucket | Leaky Bucket |
|--------|-------------|-------------|
| **Burst tolerance** | Allows controlled bursts up to capacity | Strict constant rate, no bursts |
| **Implementation** | Simple, stateless between checks | Requires queue management |
| **Our use case** | Embedding batches arrive in bursts of 32 | Would throttle even when capacity exists |

**Decision**: Token Bucket. Our dynamic batching sends chunks in bursts of 32. A leaky bucket would artificially throttle batches even when there's available capacity, reducing throughput. The token bucket allows a batch of 32 to proceed immediately if the bucket has 32+ tokens, then naturally rate-limits subsequent batches.

### 4.3 Why Redis Stack for Semantic Cache (not pgvector)?

| Factor | Redis (RediSearch) | pgvector |
|--------|-------------------|----------|
| **Latency** | Sub-millisecond (in-memory) | ~1-5ms (disk-backed) |
| **TTL support** | Native key expiration | Requires manual cleanup |
| **Data volume** | Small (only recent queries) | Large (all document chunks) |
| **Eviction** | Automatic via TTL + LRU | Manual |

**Decision**: Two-tier architecture. The semantic cache holds only recent query→result pairs (small, ephemeral) — perfect for Redis's in-memory speed. The document chunks (large, persistent) stay in PostgreSQL where they benefit from ACID guarantees and disk-backed storage.

### 4.4 Why Celery + Redis over a simpler task queue?

Alternatives considered:
- **FastAPI BackgroundTasks**: No persistence — if the server crashes, the task is lost. No visibility into task state.
- **asyncio.create_task**: Same issues. Also, CPU-bound embedding work blocks the event loop.
- **RQ (Redis Queue)**: Simpler but lacks Celery's retry policies, task routing, and monitoring ecosystem.

**Decision**: Celery. It provides `autoretry_for` with exponential backoff, `task_track_started` for progress visibility, `acks_late` for crash resilience, and a mature monitoring ecosystem (Flower). The worker pool model also isolates CPU-bound embedding work from the async API server.

### 4.5 Why 512-token chunks with 50-token overlap?

- **512 tokens**: Matches the input window of `all-MiniLM-L6-v2` (max 256 word-pieces, but ~512 whitespace tokens ≈ 256 word-pieces after tokenization). Larger chunks lose granularity; smaller chunks lose context.
- **50-token overlap**: Prevents information loss at chunk boundaries. A sentence split across two chunks will appear fully in at least one of them, ensuring retrieval doesn't miss relevant content.

---

## 5. Data Flow

### Ingestion (Write Path)

```
Client POST /api/ingest
    │
    ▼
FastAPI validates request
    │
    ▼
Celery task enqueued → 202 Accepted returned immediately
    │
    ▼
Worker dequeues task
    │
    ▼
chunk_text(doc, 512 tokens, 50 overlap)
    │
    ▼
For each micro-batch of 32 chunks:
    │
    ├── TokenBucket.acquire(32)  ← blocks if rate limited
    │
    ├── SentenceTransformer.encode(batch)  ← CPU-bound
    │
    ├── bulk INSERT into document_chunks
    │
    └── update_state(PROGRESS, {processed: N})
    │
    ▼
Task returns SUCCESS with document_id
```

### Search (Read Path)

```
Client POST /api/search
    │
    ▼
encode_single(query) → 384-dim vector
    │
    ▼
Redis FT.SEARCH KNN (k=1, cosine)
    │
    ├── similarity ≥ 0.95 → CACHE HIT → return cached results
    │
    └── similarity < 0.95 → CACHE MISS
                │
                ▼
        pgvector: SELECT ... ORDER BY embedding <=> query LIMIT top_k
                │
                ▼
        Cache results in Redis (TTL = 1 hour)
                │
                ▼
        Return ranked chunks
```

---

## 6. Scalability Considerations

### Current Design (Single-Node)
- **API**: Single FastAPI instance handles ~1000 req/s (async I/O).
- **Workers**: 2 Celery workers with prefork pool. Each processes one document at a time.
- **Bottleneck**: CPU-bound embedding on the worker.

### Scaling Strategy

| Component | How to Scale | Complexity |
|-----------|-------------|------------|
| **API** | Add more `api` replicas behind a load balancer | Low |
| **Workers** | `docker compose up --scale worker=N` | Low |
| **Redis** | Redis Cluster or Redis Sentinel | Medium |
| **PostgreSQL** | Read replicas for search, primary for writes | Medium |
| **Embeddings** | GPU-accelerated workers, or switch to API-based embedder (OpenAI) | Medium |

### What I Would Add for Production
1. **Kubernetes** — Replace Docker Compose with Helm charts for autoscaling.
2. **GPU workers** — Separate GPU-enabled worker pool for embedding generation.
3. **Batch deduplication** — Skip re-embedding documents that haven't changed (content hash).
4. **Dead letter queue** — After max retries, move failed tasks to a DLQ for manual inspection.
5. **Circuit breaker** — If PostgreSQL is down, stop accepting ingestion requests rather than queuing indefinitely.

---

## 7. Observability

### Metrics (Prometheus)
| Metric | Type | Purpose |
|--------|------|---------|
| `http_request_duration_seconds` | Histogram | API latency by endpoint |
| `documents_ingested_total` | Counter | Ingestion throughput |
| `chunks_processed_total` | Counter | Embedding throughput |
| `semantic_cache_hits_total` | Counter | Cache effectiveness |
| `semantic_cache_misses_total` | Counter | Cache miss rate |
| `embedding_duration_seconds` | Histogram | Model inference latency |
| `active_ingestion_tasks` | Gauge | Worker utilization |
| `ingestion_errors_total` | Counter | Failure rate |

### Key Dashboards (if Grafana were added)
1. **Cache Hit Ratio**: `hits / (hits + misses)` — target >60% in production.
2. **P99 Search Latency**: Should be <50ms on cache hit, <200ms on cache miss.
3. **Ingestion Throughput**: Documents/min and chunks/min.
4. **Error Rate**: `errors / total` — alert if >1%.

### Structured Logging
All logs emitted as single-line JSON for ingestion by ELK/Loki/CloudWatch:
```json
{"ts":"2024-01-15T10:30:00+00:00","level":"INFO","logger":"app.tasks","msg":"Document abc123 ingestion complete.","task_id":"celery-xyz"}
```

---

## 8. Failure Modes & Resilience

| Failure | Mitigation |
|---------|-----------|
| **Redis down** | Celery retries with exponential backoff. Semantic cache degrades gracefully (returns None on error). |
| **PostgreSQL down** | `autoretry_for=(OperationalError)` retries up to 5 times with backoff. `acks_late=True` ensures task is re-delivered if worker crashes. |
| **Worker crash mid-ingestion** | Late-ack means the task returns to the queue. Next worker picks it up. Partial inserts are idempotent (UUID primary keys). |
| **Embedding model OOM** | Token bucket caps concurrent embedding load. Micro-batching (32) keeps memory bounded. |
| **Network partition** | Health checks detect degraded state. Readiness probe fails, stopping traffic from load balancer. |
