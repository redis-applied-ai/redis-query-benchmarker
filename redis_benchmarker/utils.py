"""Utility functions for Redis Query Benchmarker."""

import time
from contextlib import contextmanager
from typing import Generator, Tuple, Any, Callable


class LatencyTimer:
    """
    A timer object that acts like a float and gets populated with latency
    when the time_operation context manager exits.
    """

    def __init__(self):
        self._latency_ms = 0.0

    def _set_latency(self, latency_ms: float) -> None:
        """Internal method to set the latency value."""
        self._latency_ms = latency_ms

    def __float__(self) -> float:
        return self._latency_ms

    def __int__(self) -> int:
        return int(self._latency_ms)

    def __str__(self) -> str:
        return str(self._latency_ms)

    def __repr__(self) -> str:
        return f"LatencyTimer({self._latency_ms}ms)"

    # Arithmetic operations to make it behave like a number
    def __add__(self, other):
        return self._latency_ms + float(other)

    def __radd__(self, other):
        return float(other) + self._latency_ms

    def __sub__(self, other):
        return self._latency_ms - float(other)

    def __rsub__(self, other):
        return float(other) - self._latency_ms

    def __mul__(self, other):
        return self._latency_ms * float(other)

    def __rmul__(self, other):
        return float(other) * self._latency_ms

    def __truediv__(self, other):
        return self._latency_ms / float(other)

    def __rtruediv__(self, other):
        return float(other) / self._latency_ms

    # Comparison operations
    def __lt__(self, other):
        return self._latency_ms < float(other)

    def __le__(self, other):
        return self._latency_ms <= float(other)

    def __eq__(self, other):
        return self._latency_ms == float(other)

    def __ne__(self, other):
        return self._latency_ms != float(other)

    def __gt__(self, other):
        return self._latency_ms > float(other)

    def __ge__(self, other):
        return self._latency_ms >= float(other)


@contextmanager
def time_operation() -> Generator[LatencyTimer, None, None]:
    """
    Context manager for timing operations.

    Usage:
        with time_operation() as latency_ms:
            # perform operation
            result = some_operation()
        # latency_ms now contains the elapsed time in milliseconds
        print(f"Operation took {latency_ms} ms")

    Returns:
        LatencyTimer object that acts like a float containing elapsed time in milliseconds
    """
    timer = LatencyTimer()
    start_time = time.perf_counter()

    try:
        yield timer
    finally:
        end_time = time.perf_counter()
        timer._set_latency((end_time - start_time) * 1000)


def time_function(func: Callable, *args, **kwargs) -> Tuple[Any, float]:
    """
    Time the execution of a function and return both result and latency.

    Args:
        func: Function to execute
        *args: Positional arguments to pass to func
        **kwargs: Keyword arguments to pass to func

    Returns:
        Tuple of (result, latency_ms)
    """
    start_time = time.perf_counter()
    result = func(*args, **kwargs)
    end_time = time.perf_counter()
    latency_ms = (end_time - start_time) * 1000
    return result, latency_ms