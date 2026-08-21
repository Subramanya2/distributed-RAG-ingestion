"""
Pydantic request / response schemas for the API layer.
"""

from pydantic import BaseModel, Field


# ── Ingestion ──────────────────────────────────────────────────────────
class IngestRequest(BaseModel):
    """Payload for POST /api/ingest."""

    text: str = Field(
        ...,
        min_length=1,
        description="Raw text document to ingest and embed.",
    )
    metadata: dict = Field(
        default_factory=dict,
        description="Arbitrary key-value metadata attached to every chunk.",
    )


class IngestResponse(BaseModel):
    """Returned immediately (202 Accepted) after an ingest request."""

    job_id: str
    status: str = "accepted"


# ── Job Status ─────────────────────────────────────────────────────────
class JobStatus(BaseModel):
    """Response for GET /api/status/{job_id}."""

    job_id: str
    status: str = Field(
        ...,
        description="PENDING | STARTED | PROGRESS | SUCCESS | FAILURE",
    )
    progress: dict = Field(
        default_factory=dict,
        description="E.g. {'total_chunks': 42, 'processed': 20}",
    )
    result: dict | None = None
    error: str | None = None


# ── Search ─────────────────────────────────────────────────────────────
class SearchRequest(BaseModel):
    """Payload for POST /api/search."""

    query: str = Field(
        ...,
        min_length=1,
        description="Natural-language search query.",
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=100,
        description="Number of nearest chunks to return.",
    )


class ChunkResult(BaseModel):
    """A single chunk returned from vector search."""

    id: str
    document_id: str
    chunk_index: int
    content: str
    score: float
    metadata: dict = {}


class SearchResponse(BaseModel):
    """Response for POST /api/search."""

    query: str
    cached: bool = False
    results: list[ChunkResult] = []
