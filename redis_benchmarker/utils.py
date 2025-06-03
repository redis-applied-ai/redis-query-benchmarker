"""Utility functions for Redis Query Benchmarker."""

import time
from contextlib import contextmanager
from typing import Generator, Tuple, Any, Callable


@contextmanager
def time_operation() -> Generator[Callable[[], float], None, None]:
    """
    Context manager for timing operations.

    Usage:
        with time_operation() as get_latency_ms:
            # perform operation
            result = some_operation()
        latency_ms = get_latency_ms()

    Returns:
        Function that returns the elapsed time in milliseconds
    """
    start_time = time.perf_counter()

    def get_latency_ms() -> float:
        end_time = time.perf_counter()
        return (end_time - start_time) * 1000

    yield get_latency_ms


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