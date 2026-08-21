"""
Load test for the RAG Ingestion Engine using Locust.

Run with:
    locust -f load_tests/locustfile.py --host=http://localhost:8000

Scenarios:
  1. IngestUser — Continuously ingests documents of varying sizes.
  2. SearchUser — Continuously queries the search endpoint.
  3. MixedUser — Realistic mix: 30% ingest, 50% search, 20% status checks.
"""

import random
import string
import uuid

from locust import HttpUser, between, task


def generate_document(word_count: int = 1000) -> str:
    """Generate a random document with realistic-ish words."""
    words = []
    vocabulary = [
        "machine", "learning", "neural", "network", "embedding",
        "vector", "database", "retrieval", "augmented", "generation",
        "transformer", "attention", "encoder", "decoder", "tokenizer",
        "semantic", "similarity", "cosine", "distance", "index",
        "query", "document", "chunk", "batch", "pipeline",
        "distributed", "cache", "redis", "postgres", "celery",
        "asynchronous", "worker", "queue", "broker", "task",
        "ingestion", "processing", "storage", "search", "ranking",
        "algorithm", "optimization", "performance", "throughput", "latency",
        "architecture", "microservice", "container", "docker", "kubernetes",
    ]
    for _ in range(word_count):
        words.append(random.choice(vocabulary))
    return " ".join(words)


class IngestUser(HttpUser):
    """Simulates users continuously ingesting documents."""

    wait_time = between(1, 3)

    @task
    def ingest_small_doc(self):
        """Ingest a small document (~500 words = ~1 chunk)."""
        self.client.post(
            "/api/ingest",
            json={
                "text": generate_document(500),
                "metadata": {"source": "load_test", "size": "small"},
            },
        )

    @task
    def ingest_medium_doc(self):
        """Ingest a medium document (~2000 words = ~4 chunks)."""
        self.client.post(
            "/api/ingest",
            json={
                "text": generate_document(2000),
                "metadata": {"source": "load_test", "size": "medium"},
            },
        )

    @task
    def ingest_large_doc(self):
        """Ingest a large document (~10000 words = ~20 chunks)."""
        self.client.post(
            "/api/ingest",
            json={
                "text": generate_document(10000),
                "metadata": {"source": "load_test", "size": "large"},
            },
        )


class SearchUser(HttpUser):
    """Simulates users performing semantic searches."""

    wait_time = between(0.5, 2)

    queries = [
        "How does the attention mechanism work in transformers?",
        "What is retrieval augmented generation?",
        "Explain cosine similarity for vector search",
        "distributed processing pipeline architecture",
        "semantic caching with redis",
        "embedding model optimization techniques",
        "neural network training best practices",
        "database indexing for vector search",
        "microservice architecture patterns",
        "asynchronous task processing with celery",
    ]

    @task(3)
    def search_varied(self):
        """Search with a random query."""
        self.client.post(
            "/api/search",
            json={"query": random.choice(self.queries), "top_k": 5},
        )

    @task(1)
    def search_repeated(self):
        """Search with the same query to test cache hit rate."""
        self.client.post(
            "/api/search",
            json={"query": "retrieval augmented generation", "top_k": 5},
        )


class MixedUser(HttpUser):
    """
    Realistic workload: mix of ingestion, search, and status checks.
    Weighted to approximate production traffic patterns.
    """

    wait_time = between(1, 5)
    job_ids: list[str] = []

    @task(3)
    def ingest(self):
        """Ingest a document and save the job_id."""
        response = self.client.post(
            "/api/ingest",
            json={
                "text": generate_document(random.choice([500, 1000, 2000])),
                "metadata": {"source": "mixed_load_test"},
            },
        )
        if response.status_code == 202:
            job_id = response.json().get("job_id")
            if job_id:
                self.job_ids.append(job_id)
                # Keep only last 50 job IDs
                if len(self.job_ids) > 50:
                    self.job_ids = self.job_ids[-50:]

    @task(5)
    def search(self):
        """Perform a search query."""
        queries = [
            "machine learning pipeline",
            "vector database optimization",
            "semantic search architecture",
        ]
        self.client.post(
            "/api/search",
            json={"query": random.choice(queries), "top_k": 5},
        )

    @task(2)
    def check_status(self):
        """Check status of a recently submitted job."""
        if self.job_ids:
            job_id = random.choice(self.job_ids)
            self.client.get(f"/api/status/{job_id}")

    @task(1)
    def health_check(self):
        """Hit the health endpoint."""
        self.client.get("/health")
