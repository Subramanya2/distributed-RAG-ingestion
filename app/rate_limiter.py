"""
Token Bucket rate-limiting utility.

Controls the throughput of embedding calls (or any resource) by
allowing at most *capacity* tokens, refilled at *refill_rate* per second.
Thread-safe via threading.Lock.
"""

import logging
import threading
import time

logger = logging.getLogger(__name__)


class TokenBucket:
    """
    Classic token-bucket algorithm for rate limiting.

    Args:
        capacity:    Maximum number of tokens the bucket can hold.
        refill_rate: Tokens added per second (fractional allowed).

    Example::

        bucket = TokenBucket(capacity=100, refill_rate=10.0)

        # Block until 32 tokens are available
        bucket.acquire(tokens=32, blocking=True)

        # Non-blocking check
        if bucket.acquire(tokens=1, blocking=False):
            do_work()
    """

    def __init__(self, capacity: int, refill_rate: float) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be > 0")
        if refill_rate <= 0:
            raise ValueError("refill_rate must be > 0")

        self._capacity = capacity
        self._refill_rate = refill_rate
        self._tokens = float(capacity)       # start full
        self._last_refill = time.monotonic()
        self._lock = threading.Lock()

    # ── Core ───────────────────────────────────────────────────────────
    def _refill(self) -> None:
        """Add tokens based on elapsed time since last refill."""
        now = time.monotonic()
        elapsed = now - self._last_refill
        added = elapsed * self._refill_rate
        self._tokens = min(self._capacity, self._tokens + added)
        self._last_refill = now

    def acquire(self, tokens: int = 1, blocking: bool = True) -> bool:
        """
        Attempt to consume *tokens* from the bucket.

        If *blocking* is True, sleeps until enough tokens accumulate.
        Returns True when the tokens are successfully consumed.
        Returns False immediately if *blocking* is False and not enough
        tokens are available.
        """
        if tokens > self._capacity:
            raise ValueError(
                f"Requested {tokens} tokens but bucket capacity is {self._capacity}."
            )

        while True:
            with self._lock:
                self._refill()
                if self._tokens >= tokens:
                    self._tokens -= tokens
                    logger.debug(
                        "TokenBucket: acquired %d tokens (%.1f remaining).",
                        tokens, self._tokens,
                    )
                    return True

                if not blocking:
                    return False

                # Calculate how long to wait for enough tokens
                deficit = tokens - self._tokens
                wait_seconds = deficit / self._refill_rate

            # Sleep *outside* the lock so other threads can refill
            logger.debug(
                "TokenBucket: waiting %.3fs for %d tokens.", wait_seconds, tokens,
            )
            time.sleep(wait_seconds)

    # ── Introspection ──────────────────────────────────────────────────
    @property
    def available(self) -> float:
        """Current approximate number of available tokens."""
        with self._lock:
            self._refill()
            return self._tokens

    def __repr__(self) -> str:
        return (
            f"TokenBucket(capacity={self._capacity}, "
            f"refill_rate={self._refill_rate}, "
            f"available={self.available:.1f})"
        )
