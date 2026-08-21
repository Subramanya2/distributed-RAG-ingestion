"""
SQLAlchemy ORM model for document chunks stored with pgvector embeddings.
"""

import uuid
from datetime import datetime, timezone

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    Column,
    DateTime,
    Integer,
    String,
    Text,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import DeclarativeBase

from app.config import settings


class Base(DeclarativeBase):
    """Declarative base for all ORM models."""
    pass


class DocumentChunk(Base):
    """
    Stores one chunk of a larger ingested document alongside its
    vector embedding for ANN retrieval via pgvector HNSW.
    """

    __tablename__ = "document_chunks"

    id = Column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    document_id = Column(
        String(64),
        nullable=False,
        index=True,
        comment="Groups chunks belonging to the same source document.",
    )
    chunk_index = Column(
        Integer,
        nullable=False,
        comment="Positional index of this chunk within its document.",
    )
    content = Column(
        Text,
        nullable=False,
        comment="Raw text content of this chunk.",
    )
    embedding = Column(
        Vector(settings.EMBEDDING_DIM),
        comment="Dense vector embedding (384-dim for MiniLM-L6-v2).",
    )
    metadata_ = Column(
        "metadata",
        JSONB,
        default=dict,
        server_default="{}",
        comment="Arbitrary key-value metadata supplied at ingestion time.",
    )
    created_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    def __repr__(self) -> str:
        return (
            f"<DocumentChunk(id={self.id!s:.8}, "
            f"doc={self.document_id}, idx={self.chunk_index})>"
        )
