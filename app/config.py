"""
Centralized configuration loaded from environment variables / .env file.
"""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # ── PostgreSQL ──
    DATABASE_URL: str = "postgresql://raguser:ragpassword@db:5432/ragdb"

    # ── Redis ──
    REDIS_URL: str = "redis://redis:6379/0"

    # ── Embedding Model ──
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    EMBEDDING_DIM: int = 384

    # ── Semantic Cache ──
    CACHE_TTL: int = 3600  # seconds
    CACHE_SIMILARITY_THRESHOLD: float = 0.95

    # ── Chunking ──
    CHUNK_SIZE: int = 512     # approximate token count per chunk
    CHUNK_OVERLAP: int = 50   # token overlap between consecutive chunks

    # ── Dynamic Batching ──
    BATCH_SIZE: int = 32

    # ── Rate Limiter (Token Bucket) ──
    RATE_LIMIT_CAPACITY: int = 100
    RATE_LIMIT_REFILL_RATE: float = 10.0

    model_config = {"env_file": ".env", "extra": "ignore"}


settings = Settings()
