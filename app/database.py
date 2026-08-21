"""
Database engine, session factory, and initialization logic.

Creates the pgvector extension, ORM tables, and HNSW index on startup.
"""

import logging

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

from app.config import settings
from app.models import Base

logger = logging.getLogger(__name__)

# ── Engine & Session ───────────────────────────────────────────────────
engine = create_engine(
    settings.DATABASE_URL,
    pool_size=10,
    max_overflow=20,
    pool_pre_ping=True,
)

SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)


def get_db():
    """FastAPI dependency — yields a scoped DB session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db() -> None:
    """
    Idempotent database bootstrap:
    1. Ensures the pgvector extension exists.
    2. Creates all ORM tables.
    3. Creates an HNSW index for fast ANN cosine-similarity search.
    """
    with engine.begin() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))

    # Create tables from ORM metadata
    Base.metadata.create_all(bind=engine)

    # Create HNSW index (idempotent — IF NOT EXISTS)
    hnsw_ddl = text(
        """
        CREATE INDEX IF NOT EXISTS idx_chunks_embedding_hnsw
        ON document_chunks
        USING hnsw (embedding vector_cosine_ops)
        WITH (m = 16, ef_construction = 200);
        """
    )
    with engine.begin() as conn:
        conn.execute(hnsw_ddl)

    logger.info("Database initialized: tables + HNSW index ready.")
