"""
Unit tests for the FastAPI endpoints.

Uses unittest.mock to stub out Celery, Redis, and DB dependencies
so these tests run without any infrastructure.
"""

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    """Create a test client with mocked dependencies."""
    # Mock init_db and SemanticCache before importing the app
    with patch("app.main.init_db"), \
         patch("app.main.SemanticCache") as mock_cache_cls:
        mock_cache_cls.return_value = MagicMock()
        from app.main import app
        with TestClient(app) as c:
            yield c


class TestHealthEndpoints:
    """Tests for /health and /health/ready."""

    def test_liveness(self, client):
        """GET /health always returns 200."""
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"

    @patch("app.main.celery_app")
    @patch("app.main.redis")
    @patch("app.main.engine")
    def test_readiness_all_healthy(self, mock_engine, mock_redis, mock_celery, client):
        """GET /health/ready returns 'ready' when all services are up."""
        # Mock DB connection
        mock_conn = MagicMock()
        mock_engine.connect.return_value.__enter__ = lambda s: mock_conn
        mock_engine.connect.return_value.__exit__ = MagicMock(return_value=False)

        # Mock Redis ping
        mock_redis_client = MagicMock()
        mock_redis.from_url.return_value = mock_redis_client

        # Mock Celery inspector
        mock_inspector = MagicMock()
        mock_inspector.ping.return_value = {"worker1": {"ok": "pong"}}
        mock_celery.control.inspect.return_value = mock_inspector

        response = client.get("/health/ready")
        assert response.status_code == 200
        data = response.json()
        assert data["services"]["db"] == "ok"
        assert data["services"]["redis"] == "ok"


class TestIngestEndpoint:
    """Tests for POST /api/ingest."""

    def test_ingest_returns_202(self, client):
        """POST /api/ingest returns 202 with a job_id."""
        with patch("app.tasks.ingest_document") as mock_task:
            mock_task.delay.return_value = MagicMock(id="test-job-123")

            response = client.post(
                "/api/ingest",
                json={"text": "This is a test document.", "metadata": {"source": "test"}},
            )
            assert response.status_code == 202
            data = response.json()
            assert "job_id" in data
            assert data["status"] == "accepted"

    def test_ingest_empty_text_rejected(self, client):
        """POST /api/ingest with empty text returns 422."""
        response = client.post("/api/ingest", json={"text": ""})
        assert response.status_code == 422


class TestStatusEndpoint:
    """Tests for GET /api/status/{job_id}."""

    @patch("app.main.AsyncResult")
    def test_status_pending(self, mock_async_result, client):
        """Status endpoint returns PENDING for unknown jobs."""
        mock_result = MagicMock()
        mock_result.state = "PENDING"
        mock_result.info = None
        mock_result.result = None
        mock_async_result.return_value = mock_result

        response = client.get("/api/status/some-job-id")
        assert response.status_code == 200
        assert response.json()["status"] == "PENDING"

    @patch("app.main.AsyncResult")
    def test_status_success(self, mock_async_result, client):
        """Status endpoint returns result on SUCCESS."""
        mock_result = MagicMock()
        mock_result.state = "SUCCESS"
        mock_result.result = {"document_id": "abc", "total_chunks": 5, "status": "completed"}
        mock_async_result.return_value = mock_result

        response = client.get("/api/status/some-job-id")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "SUCCESS"
        assert data["result"]["total_chunks"] == 5


class TestSearchEndpoint:
    """Tests for POST /api/search."""

    @patch("app.main.encode_single")
    def test_search_cache_hit(self, mock_encode, client):
        """Search returns cached results on semantic cache hit."""
        mock_encode.return_value = [0.1] * 384

        # Configure the semantic cache mock on the app
        from app.main import app
        mock_cache = app.state.semantic_cache
        mock_cache.get.return_value = [
            {
                "id": "chunk-1",
                "document_id": "doc-1",
                "chunk_index": 0,
                "content": "cached content",
                "score": 0.98,
                "metadata": {},
            }
        ]

        response = client.post(
            "/api/search",
            json={"query": "test query", "top_k": 5},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["cached"] is True
        assert len(data["results"]) == 1

    @patch("app.main.SessionLocal")
    @patch("app.main.encode_single")
    def test_search_cache_miss_fallback(self, mock_encode, mock_session, client):
        """Search falls back to pgvector on cache miss."""
        mock_encode.return_value = [0.1] * 384

        from app.main import app
        mock_cache = app.state.semantic_cache
        mock_cache.get.return_value = None  # cache miss

        # Mock DB query returning empty results
        mock_db = MagicMock()
        mock_db.query.return_value.order_by.return_value.limit.return_value.all.return_value = []
        mock_session.return_value = mock_db

        response = client.post(
            "/api/search",
            json={"query": "test query", "top_k": 5},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["cached"] is False

    def test_search_empty_query_rejected(self, client):
        """POST /api/search with empty query returns 422."""
        response = client.post("/api/search", json={"query": ""})
        assert response.status_code == 422


class TestMetricsEndpoint:
    """Tests for GET /metrics."""

    def test_metrics_returns_prometheus_format(self, client):
        """Metrics endpoint returns Prometheus text format."""
        response = client.get("/metrics")
        assert response.status_code == 200
        assert "http_requests_total" in response.text
        assert "text/plain" in response.headers.get("content-type", "") or \
               "text/plain" in response.headers.get("Content-Type", "") or \
               "openmetrics" in response.headers.get("content-type", "")
