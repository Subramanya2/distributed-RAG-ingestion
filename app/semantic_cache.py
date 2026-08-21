"""
Semantic Cache backed by Redis Stack (RediSearch + RedisJSON).

Stores query embeddings alongside their results. On a new query, performs
a KNN vector search; if cosine similarity ≥ threshold, returns the cached
result — bypassing the vector DB and/or LLM entirely.
"""

import json
import logging
import time
import uuid
from typing import Any

import numpy as np
import redis
from redis.commands.search.field import TextField, VectorField
try:
    from redis.commands.search.index_definition import IndexDefinition, IndexType
except ImportError:
    from redis.commands.search.indexDefinition import IndexDefinition, IndexType
from redis.commands.search.query import Query

from app.config import settings

logger = logging.getLogger(__name__)

# Redis key prefix for all cache entries
_KEY_PREFIX = "semcache:"
_INDEX_NAME = "idx:semcache"


class SemanticCache:
    """
    Redis-backed semantic cache with RediSearch vector indexing.

    Usage::

        cache = SemanticCache()
        hit = cache.get(query_embedding)
        if hit is not None:
            return hit  # cache hit — skip expensive search/LLM
        ...
        cache.set(query_text, query_embedding, result_payload)
    """

    def __init__(
        self,
        redis_url: str = settings.REDIS_URL,
        dim: int = settings.EMBEDDING_DIM,
        threshold: float = settings.CACHE_SIMILARITY_THRESHOLD,
        ttl: int = settings.CACHE_TTL,
    ):
        self._client = redis.from_url(redis_url)
        self._dim = dim
        self._threshold = threshold
        self._ttl = ttl
        self._ensure_index()

    # ── Index Bootstrap ────────────────────────────────────────────────
    def _ensure_index(self) -> None:
        """Create the RediSearch vector index if it doesn't already exist."""
        try:
            self._client.ft(_INDEX_NAME).info()
            logger.debug("RediSearch index '%s' already exists.", _INDEX_NAME)
        except redis.ResponseError:
            logger.info("Creating RediSearch index '%s'...", _INDEX_NAME)
            schema = (
                TextField("$.query", as_name="query"),
                VectorField(
                    "$.embedding",
                    "HNSW",
                    {
                        "TYPE": "FLOAT32",
                        "DIM": self._dim,
                        "DISTANCE_METRIC": "COSINE",
                        "M": 16,
                        "EF_CONSTRUCTION": 200,
                    },
                    as_name="embedding",
                ),
            )
            definition = IndexDefinition(
                prefix=[_KEY_PREFIX],
                index_type=IndexType.JSON,
            )
            self._client.ft(_INDEX_NAME).create_index(
                schema,
                definition=definition,
            )
            logger.info("RediSearch index '%s' created.", _INDEX_NAME)

    # ── Write ──────────────────────────────────────────────────────────
    def set(
        self,
        query: str,
        embedding: list[float],
        result: Any,
    ) -> str:
        """
        Cache a query → result pair alongside its embedding vector.

        Returns the cache entry key.
        """
        key = f"{_KEY_PREFIX}{uuid.uuid4().hex}"
        payload = {
            "query": query,
            "embedding": embedding,
            "result": result,
            "timestamp": time.time(),
        }
        self._client.json().set(key, "$", payload)
        if self._ttl > 0:
            self._client.expire(key, self._ttl)
        logger.debug("Cached query under key %s (TTL=%ds).", key, self._ttl)
        return key

    # ── Read (KNN search) ─────────────────────────────────────────────
    def get(
        self,
        query_embedding: list[float],
        k: int = 1,
    ) -> list[dict] | None:
        """
        Search the cache for the nearest stored embedding.

        If cosine similarity ≥ threshold → return the cached result list/dict.
        Otherwise → return None (cache miss).
        """
        blob = np.array(query_embedding, dtype=np.float32).tobytes()

        q = (
            Query(f"(*)=>[KNN {k} @embedding $vec AS score]")
            .sort_by("score")
            .return_fields("score")
            .paging(0, k)
            .dialect(2)
        )

        try:
            results = self._client.ft(_INDEX_NAME).search(
                q, query_params={"vec": blob}
            )
        except redis.ResponseError as exc:
            logger.warning("Semantic cache search failed: %s", exc)
            return None

        if not results.docs:
            return None

        top = results.docs[0]
        # RediSearch COSINE distance ∈ [0, 2]; similarity = 1 - distance
        distance = float(top.score)
        similarity = 1.0 - distance

        logger.debug(
            "Cache candidate: similarity=%.4f, threshold=%.4f",
            similarity,
            self._threshold,
        )

        if similarity >= self._threshold:
            logger.info("Semantic cache HIT (similarity=%.4f).", similarity)
            doc_data = self._client.json().get(top.id)
            if doc_data and "result" in doc_data:
                cached_res = doc_data["result"]
                if isinstance(cached_res, str):
                    try:
                        return json.loads(cached_res)
                    except Exception:
                        return cached_res
                return cached_res

        logger.debug("Semantic cache MISS (similarity=%.4f < %.4f).", similarity, self._threshold)
        return None
