"""
Unit tests for the text chunker utility.
"""

import pytest

from app.tasks import chunk_text


class TestChunkText:
    """Tests for chunk_text() function."""

    def test_basic_chunking(self):
        """Text is split into chunks of the expected size."""
        words = " ".join(f"word{i}" for i in range(100))
        chunks = chunk_text(words, chunk_size=20, overlap=0)
        assert len(chunks) == 5  # 100 words / 20 per chunk
        assert all(len(c.split()) == 20 for c in chunks)

    def test_overlap_creates_shared_words(self):
        """Overlap tokens appear in both adjacent chunks."""
        words = " ".join(f"w{i}" for i in range(30))
        chunks = chunk_text(words, chunk_size=20, overlap=5)

        # First chunk: w0..w19, second chunk: w15..w29 (overlap of 5)
        first_words = chunks[0].split()
        second_words = chunks[1].split()
        overlap_words = set(first_words) & set(second_words)
        assert len(overlap_words) == 5

    def test_small_text_single_chunk(self):
        """Text shorter than chunk_size produces a single chunk."""
        chunks = chunk_text("hello world", chunk_size=512, overlap=50)
        assert len(chunks) == 1
        assert chunks[0] == "hello world"

    def test_empty_string(self):
        """Empty input produces no chunks."""
        assert chunk_text("") == []
        assert chunk_text("   ") == []

    def test_exact_chunk_size(self):
        """Text with exactly chunk_size words produces one chunk."""
        words = " ".join(f"w{i}" for i in range(20))
        chunks = chunk_text(words, chunk_size=20, overlap=5)
        assert len(chunks) == 1

    def test_overlap_larger_than_remainder(self):
        """Handles edge case where overlap > remaining words gracefully."""
        words = " ".join(f"w{i}" for i in range(25))
        chunks = chunk_text(words, chunk_size=20, overlap=10)
        assert len(chunks) == 2
        # Second chunk should contain words 10..24
        assert len(chunks[1].split()) == 15
