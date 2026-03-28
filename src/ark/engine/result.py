"""Rust-style Result type (Ok / Error) with combinators."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, TypeVar, Union
from collections.abc import Callable

T = TypeVar("T")
U = TypeVar("U")
E = TypeVar("E")
F = TypeVar("F")


@dataclass(slots=True)
class Ok(Generic[T]):
    value: T

    def is_ok(self) -> bool:
        return True

    def is_err(self) -> bool:
        return False

    def unwrap(self) -> T:
        return self.value

    def unwrap_or(self, default: T) -> T:  # type: ignore[override]
        return self.value

    def map(self, fn: Callable[[T], U]) -> Ok[U]:
        return Ok(fn(self.value))

    def inspect(self, fn: Callable[[T], object]) -> Ok[T]:
        fn(self.value)
        return self

    def map_err(self, fn: Callable[[object], F]) -> Ok[T]:  # type: ignore[override]
        return self  # type: ignore[return-value]


@dataclass(slots=True)
class Error(Generic[E]):
    error: E

    def is_ok(self) -> bool:
        return False

    def is_err(self) -> bool:
        return True

    def unwrap(self) -> E:
        raise RuntimeError(f"Called unwrap on Error: {self.error}")

    def unwrap_or(self, default: T) -> T:  # type: ignore[type-var]
        return default

    def map(self, fn: Callable[[object], U]) -> Error[E]:  # type: ignore[override]
        return self  # type: ignore[return-value]

    def inspect(self, fn: Callable[[object], object]) -> Error[E]:
        return self

    def map_err(self, fn: Callable[[E], F]) -> Error[F]:
        return Error(fn(self.error))


Result = Union[Ok[T], Error[E]]
