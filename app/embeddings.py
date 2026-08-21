"""
Embedding model wrapper — singleton SentenceTransformer instance.

Provides a thin interface for encoding text into dense vectors.
The model is loaded once and reused across all calls.
"""

import logging
from functools import lru_cache

import numpy as np
from sentence_transformers import SentenceTransformer

from app.config import settings

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _load_model() -> SentenceTransformer:
    """Load the embedding model once and cache it in memory."""
    logger.info("Loading embedding model: %s", settings.EMBEDDING_MODEL)
    model = SentenceTransformer(settings.EMBEDDING_MODEL)
    logger.info("Embedding model loaded (%d dimensions).", settings.EMBEDDING_DIM)
    return model


from app.metrics import track_embedding_time


@track_embedding_time
def encode(texts: list[str]) -> np.ndarray:
    """
    Encode a list of text strings into dense vectors.

    Args:
        texts: List of strings to embed.

    Returns:
        np.ndarray of shape (len(texts), EMBEDDING_DIM).
    """
    model = _load_model()
    embeddings: np.ndarray = model.encode(
        texts,
        show_progress_bar=False,
        normalize_embeddings=True,  # unit-norm for cosine similarity
    )
    return embeddings


def encode_single(text: str) -> list[float]:
    """Convenience wrapper — encode a single string and return a flat list."""
    return encode([text])[0].tolist()
