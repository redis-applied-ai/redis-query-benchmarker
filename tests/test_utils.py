"""Tests for timing utilities."""

import time
import pytest
from redis_benchmarker.utils import time_operation, time_function


class TestTimingUtilities:
    """Test timing utility functions."""

    def test_time_operation_context_manager(self):
        """Test the time_operation context manager."""
        sleep_duration = 0.1

        with time_operation() as latency_ms:
            time.sleep(sleep_duration)

        # Should be approximately 100ms (with some tolerance for system variance)
        assert 90 <= latency_ms <= 150, f"Expected ~100ms, got {latency_ms}ms"

    def test_time_operation_multiple_calls(self):
        """Test that LatencyTimer behaves like a number and can be used multiple times."""
        with time_operation() as latency_ms:
            time.sleep(0.1)

        # Test that we can use the timer value multiple times
        first_access = float(latency_ms)
        second_access = float(latency_ms)

        # Both accesses should return the same value
        assert first_access == second_access
        assert 90 <= first_access <= 150  # Should be around 100ms

    def test_time_function_wrapper(self):
        """Test the time_function wrapper."""
        def test_func(duration):
            time.sleep(duration)
            return "result"

        result, latency_ms = time_function(test_func, 0.1)

        assert result == "result"
        assert 90 <= latency_ms <= 150, f"Expected ~100ms, got {latency_ms}ms"

    def test_time_function_with_kwargs(self):
        """Test time_function with keyword arguments."""
        def test_func(multiplier, base=10):
            time.sleep(0.05)
            return multiplier * base

        result, latency_ms = time_function(test_func, 5, base=20)

        assert result == 100  # 5 * 20
        assert 40 <= latency_ms <= 80, f"Expected ~50ms, got {latency_ms}ms"

    def test_time_function_with_exception(self):
        """Test time_function when the wrapped function raises an exception."""
        def failing_func():
            time.sleep(0.05)
            raise ValueError("Test error")

        with pytest.raises(ValueError, match="Test error"):
            time_function(failing_func)

    def test_timing_accuracy(self):
        """Test that timing measurements are reasonably accurate."""
        # Test multiple durations to ensure accuracy
        test_durations = [0.01, 0.05, 0.1]
        tolerance = 0.02  # 20ms tolerance

        for duration in test_durations:
            with time_operation() as latency_ms:
                time.sleep(duration)

            expected_ms = duration * 1000

            assert abs(latency_ms - expected_ms) <= (tolerance * 1000), \
                f"Duration {duration}s: expected ~{expected_ms}ms, got {latency_ms}ms"

    def test_latency_timer_numeric_behavior(self):
        """Test that LatencyTimer behaves like a number."""
        with time_operation() as latency_ms:
            time.sleep(0.05)

        # Test conversion to float
        assert isinstance(float(latency_ms), float)

        # Test conversion to int
        assert isinstance(int(latency_ms), int)

        # Test arithmetic operations
        doubled = latency_ms * 2
        assert doubled == float(latency_ms) * 2

        # Test comparison operations
        assert latency_ms > 0
        assert latency_ms >= 0
        assert latency_ms == latency_ms

        # Test string representation
        assert str(latency_ms) == str(float(latency_ms))