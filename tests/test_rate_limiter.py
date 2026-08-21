"""
Unit tests for the TokenBucket rate limiter.
"""

import time
import threading

import pytest

from app.rate_limiter import TokenBucket


class TestTokenBucket:
    """Tests for TokenBucket rate limiter."""

    def test_initial_capacity_full(self):
        """Bucket starts at full capacity."""
        bucket = TokenBucket(capacity=10, refill_rate=1.0)
        assert bucket.available == 10.0

    def test_acquire_consumes_tokens(self):
        """Acquiring tokens reduces available count."""
        bucket = TokenBucket(capacity=10, refill_rate=1.0)
        assert bucket.acquire(5, blocking=False)
        assert bucket.available == pytest.approx(5.0, abs=0.5)

    def test_acquire_fails_when_empty(self):
        """Non-blocking acquire returns False when insufficient tokens."""
        bucket = TokenBucket(capacity=5, refill_rate=1.0)
        assert bucket.acquire(5, blocking=False)
        assert not bucket.acquire(1, blocking=False)

    def test_refill_over_time(self):
        """Tokens refill after waiting."""
        bucket = TokenBucket(capacity=10, refill_rate=100.0)  # fast refill
        bucket.acquire(10, blocking=False)
        time.sleep(0.1)  # ~10 tokens should refill
        assert bucket.available >= 5.0  # conservative check

    def test_capacity_not_exceeded(self):
        """Refill never exceeds capacity."""
        bucket = TokenBucket(capacity=5, refill_rate=1000.0)
        time.sleep(0.1)
        assert bucket.available <= 5.0

    def test_blocking_acquire_waits(self):
        """Blocking acquire sleeps until tokens are available."""
        bucket = TokenBucket(capacity=5, refill_rate=50.0)  # refills fast
        bucket.acquire(5, blocking=False)  # drain
        start = time.monotonic()
        assert bucket.acquire(1, blocking=True)
        elapsed = time.monotonic() - start
        assert elapsed < 1.0  # should resolve quickly with fast refill

    def test_acquire_more_than_capacity_raises(self):
        """Requesting more tokens than capacity raises ValueError."""
        bucket = TokenBucket(capacity=5, refill_rate=1.0)
        with pytest.raises(ValueError, match="capacity"):
            bucket.acquire(10)

    def test_invalid_capacity_raises(self):
        """Zero or negative capacity raises ValueError."""
        with pytest.raises(ValueError):
            TokenBucket(capacity=0, refill_rate=1.0)
        with pytest.raises(ValueError):
            TokenBucket(capacity=-1, refill_rate=1.0)

    def test_invalid_refill_rate_raises(self):
        """Zero or negative refill_rate raises ValueError."""
        with pytest.raises(ValueError):
            TokenBucket(capacity=10, refill_rate=0)
        with pytest.raises(ValueError):
            TokenBucket(capacity=10, refill_rate=-5)

    def test_thread_safety(self):
        """Concurrent access doesn't corrupt the token count."""
        bucket = TokenBucket(capacity=100, refill_rate=0.0001)  # near-zero refill
        results = []

        def consumer():
            success = bucket.acquire(1, blocking=False)
            results.append(success)

        threads = [threading.Thread(target=consumer) for _ in range(100)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All 100 should succeed (capacity=100) and bucket should be near-empty
        assert sum(results) == 100
        assert bucket.available < 1.0

    def test_repr(self):
        """__repr__ includes useful debug info."""
        bucket = TokenBucket(capacity=10, refill_rate=2.0)
        r = repr(bucket)
        assert "capacity=10" in r
        assert "refill_rate=2.0" in r
