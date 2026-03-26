"""Circuit breaker for embedding providers."""

from __future__ import annotations

import asyncio
import time

from ark.engine.result import Error, Ok, Result
from ark.engine.embed import Embedding
from ark.engine.types import IndexErr

_DEFAULT_TIMEOUT = 10.0
_DEFAULT_FAILURE_THRESHOLD = 3
_DEFAULT_COOLDOWN = 60.0


class _State:
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreakerEmbedding:
    __slots__ = (
        "_inner", "_timeout", "_failure_threshold", "_cooldown",
        "_state", "_consecutive_failures", "_opened_at",
    )

    def __init__(
        self,
        inner: Embedding,
        timeout: float = _DEFAULT_TIMEOUT,
        failure_threshold: int = _DEFAULT_FAILURE_THRESHOLD,
        cooldown: float = _DEFAULT_COOLDOWN,
    ) -> None:
        self._inner = inner
        self._timeout = timeout
        self._failure_threshold = failure_threshold
        self._cooldown = cooldown
        self._state = _State.CLOSED
        self._consecutive_failures = 0
        self._opened_at = 0.0

    @property
    def dims(self) -> int:
        return self._inner.dims

    async def embed(self, text: str) -> Result[list[float], IndexErr]:
        if self._state == _State.OPEN:
            if time.monotonic() - self._opened_at >= self._cooldown:
                self._state = _State.HALF_OPEN
            else:
                return Error(IndexErr(code="circuit_open", message="Embedding circuit open"))

        try:
            result = await asyncio.wait_for(self._inner.embed(text), timeout=self._timeout)
        except asyncio.TimeoutError:
            return self._record_failure(f"Embedding timed out after {self._timeout}s")

        match result:
            case Ok(_):
                self._record_success()
                return result
            case Error(err):
                return self._record_failure(err.message)

    def _record_success(self) -> None:
        self._consecutive_failures = 0
        self._state = _State.CLOSED

    def _record_failure(self, message: str) -> Result[list[float], IndexErr]:
        self._consecutive_failures += 1
        if self._state == _State.HALF_OPEN or self._consecutive_failures >= self._failure_threshold:
            self._state = _State.OPEN
            self._opened_at = time.monotonic()
        return Error(IndexErr(code="embed_error", message=message))
