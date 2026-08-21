# 🚀 RAG Ingestion Engine — Distributed High-Throughput Pipeline

A production-grade, distributed backend system that ingests raw text documents, asynchronously chunks and embeds them via Celery workers, stores vectors in PostgreSQL/pgvector with HNSW indexing, and fronts the system with a FastAPI gateway featuring Redis-backed semantic caching, token-bucket rate limiting, and Prometheus observability.

> **This is the Retrieval (R) layer of a RAG system** — it makes documents semantically searchable. Plug an LLM into the search results for the Generation (G) layer.

---

## Architecture

```
┌─────────────┐         ┌─────────────────────────────────┐
│   Client    │────────▶│       FastAPI Gateway            │
│             │◀────────│  (port 8000)                     │
└─────────────┘         │                                  │
                        │  POST /api/ingest  → enqueue     │
                        │  POST /api/search  → cache + ANN │
                        │  GET  /api/status  → poll        │
                        │  GET  /metrics     → Prometheus  │
                        └──────┬───────────────┬───────────┘
                               │               │
                    ┌──────────▼──┐    ┌───────▼────────┐
                    │    Redis    │    │   PostgreSQL   │
                    │  Stack      │    │  + pgvector    │
                    │             │    │                │
                    │ • Broker    │    │ • HNSW index   │
                    │ • Results   │    │ • 384-dim vecs │
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

## Tech Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **API** | FastAPI | Async HTTP gateway |
| **Task Queue** | Celery + Redis | Async document processing |
| **Vector DB** | PostgreSQL + pgvector | HNSW-indexed ANN search |
| **Semantic Cache** | Redis Stack (RediSearch) | Sub-ms cached query responses |
| **Embeddings** | sentence-transformers (MiniLM-L6-v2) | 384-dim dense vectors |
| **Rate Limiter** | Custom Token Bucket | Controlled embedding throughput |
| **Metrics** | Prometheus | Observability |
| **Container Orchestration** | Docker Compose | 4-service stack |
| **CI/CD** | GitHub Actions | Lint, test, build, integration |

---

## Quick Start

### Prerequisites
- [Docker Desktop](https://docs.docker.com/desktop/) (v20+)

### 1. Clone & Configure

```bash
git clone <your-repo-url>
cd system-design
cp .env.example .env
```

### 2. Build & Run

```bash
# Start all services in the background (detached mode)
docker compose up -d

# OR start in the foreground to see live streaming logs
docker compose up
```

First build takes ~3–4 minutes (downloads PyTorch CPU + embedding model). Subsequent starts take ~2 seconds.

### 3. Docker Lifecycle & Management Commands

| Action | Command |
|---|---|
| **Start in background** | `docker compose up -d` |
| **View live logs** | `docker compose logs -f` |
| **View logs for specific service** | `docker compose logs -f api` *(or `worker`, `db`, `redis`)* |
| **Restart a service** | `docker compose restart api` |
| **Run unit tests inside container** | `docker compose exec api pytest tests/ -v` |
| **Stop services (keep data)** | `docker compose stop` |
| **Shutdown & remove containers** | `docker compose down` |
| **Full clean teardown (wipe DB & cache data)** | `docker compose down -v` |

### 4. Verify

```bash
# Health check
curl http://localhost:8000/health

# Full readiness (checks DB + Redis + Celery connectivity)
curl http://localhost:8000/health/ready
```

---

## API Reference

Interactive docs available at **http://localhost:8000/docs** (Swagger UI) and **http://localhost:8000/redoc** (ReDoc).

### `POST /api/ingest` — Ingest a Document

```bash
curl -X POST http://localhost:8000/api/ingest \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Your long document text here. Machine learning is a subset of AI...",
    "metadata": {"source": "research-paper", "author": "Jane Doe"}
  }'
```

**Response (202 Accepted):**
```json
{
  "job_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "status": "accepted"
}
```

### `GET /api/status/{job_id}` — Check Ingestion Progress

```bash
curl http://localhost:8000/api/status/a1b2c3d4-e5f6-7890-abcd-ef1234567890
```

**Response:**
```json
{
  "job_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "status": "PROGRESS",
  "progress": {
    "document_id": "abc123",
    "total_chunks": 42,
    "processed": 32
  }
}
```

### `POST /api/search` — Semantic Search

```bash
curl -X POST http://localhost:8000/api/search \
  -H "Content-Type: application/json" \
  -d '{"query": "How does attention work in transformers?", "top_k": 5}'
```

**Response:**
```json
{
  "query": "How does attention work in transformers?",
  "cached": false,
  "results": [
    {
      "id": "chunk-uuid",
      "document_id": "doc-uuid",
      "chunk_index": 3,
      "content": "The attention mechanism computes weighted...",
      "score": 0.8723,
      "metadata": {"source": "research-paper"}
    }
  ]
}
```

A second identical (or semantically similar) query returns `"cached": true` with sub-millisecond latency.

### `GET /metrics` — Prometheus Metrics

```bash
curl http://localhost:8000/metrics
```

Returns Prometheus text format with: `http_request_duration_seconds`, `documents_ingested_total`, `semantic_cache_hits_total`, `embedding_duration_seconds`, and more.

---

## Project Structure

```
├── .env.example                # Environment template
├── .github/workflows/ci.yml   # CI pipeline (lint, test, build, integration)
├── Dockerfile                  # CPU-optimized multi-purpose image
├── docker-compose.yml          # 4-service stack with health checks
├── requirements.txt            # Python dependencies
│
├── app/
│   ├── main.py                 # FastAPI gateway (5 endpoints + metrics middleware)
│   ├── config.py               # Pydantic Settings (env-driven)
│   ├── database.py             # SQLAlchemy engine + HNSW index DDL
│   ├── models.py               # DocumentChunk ORM (pgvector)
│   ├── schemas.py              # Pydantic request/response models
│   ├── embeddings.py           # SentenceTransformer singleton + metrics
│   ├── semantic_cache.py       # Redis + RediSearch vector cache
│   ├── celery_app.py           # Celery factory (Redis broker/backend)
│   ├── tasks.py                # Ingestion task (retry, batching, rate-limit)
│   ├── rate_limiter.py         # Token Bucket algorithm
│   ├── metrics.py              # Prometheus metric definitions
│   └── logging_config.py       # Structured JSON logging
│
├── tests/
│   ├── test_api.py             # Endpoint tests (health, ingest, search, metrics)
│   ├── test_chunker.py         # Text chunking unit tests
│   └── test_rate_limiter.py    # Token bucket unit tests (incl. thread safety)
│
├── load_tests/
│   └── locustfile.py           # Load test (ingest, search, mixed users)
│
├── docs/
│   └── SYSTEM_DESIGN.md        # Architecture decisions & trade-offs
│
└── scripts/
    └── init_db.sql             # pgvector extension bootstrap
```

---

## Testing

### Unit Tests

```bash
pip install pytest pytest-cov
pytest tests/ -v --cov=app --cov-report=term-missing
```

### Load Tests

```bash
pip install locust
locust -f load_tests/locustfile.py --host=http://localhost:8000
```

Then open http://localhost:8089 for the Locust web UI.

---

## Key Design Decisions

| Decision | Choice | Why |
|----------|--------|-----|
| **Vector index** | HNSW over IVFFlat | Supports continuous inserts without rebuild |
| **Rate limiter** | Token Bucket over Leaky Bucket | Allows controlled bursts for batch processing |
| **Semantic cache** | Redis over pgvector | Sub-ms latency for ephemeral query cache |
| **Task queue** | Celery over BackgroundTasks | Crash resilience, retry policies, monitoring |
| **PyTorch** | CPU-only | Keeps image at ~1.5GB vs ~4GB with CUDA |

See [docs/SYSTEM_DESIGN.md](docs/SYSTEM_DESIGN.md) for the full rationale with trade-off analysis.

---

## Configuration

All config is driven by environment variables (see `.env.example`):

| Variable | Default | Description |
|----------|---------|-------------|
| `CHUNK_SIZE` | `512` | Tokens per chunk |
| `CHUNK_OVERLAP` | `50` | Overlap between chunks |
| `BATCH_SIZE` | `32` | Chunks per embedding batch |
| `CACHE_SIMILARITY_THRESHOLD` | `0.95` | Cosine sim for cache hit |
| `CACHE_TTL` | `3600` | Cache entry lifetime (sec) |
| `RATE_LIMIT_CAPACITY` | `100` | Token bucket max burst |
| `RATE_LIMIT_REFILL_RATE` | `10.0` | Tokens refilled per second |

---

## License

MIT
