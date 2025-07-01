"""Tests for benchmarker module."""

import pytest
from unittest.mock import Mock, MagicMock, patch
import tempfile
import json
import csv
import time

from redis_benchmarker.config import BenchmarkConfig
from redis_benchmarker.benchmarker import RedisBenchmarker, BenchmarkResults
from redis_benchmarker.executors import BaseQueryExecutor


class MockExecutor(BaseQueryExecutor):
    """Mock executor for testing."""

    def __init__(self, config, latency=100.0, should_error=False):
        super().__init__(config)
        self.latency = latency
        self.should_error = should_error
        self.call_count = 0

    def execute_query(self, redis_client):
        self.call_count += 1
        if self.should_error:
            raise Exception("Mock error")

        return {
            "result": {"doc": f"result_{self.call_count}"},
            "latency_ms": self.latency,
            "metadata": {"mock": True}
        }


class TestBenchmarkResults:
    """Test benchmark results container."""

    def test_qps_calculation(self):
        """Test QPS calculation."""
        config = BenchmarkConfig()
        results = BenchmarkResults(
            total_requests=1000,
            successful_requests=800,
            failed_requests=200,
            total_time=10.0,
            latencies=[100.0] * 800,
            errors=[],
            metadata={},
            config=config
        )

        assert results.qps == 80.0  # 800 / 10

    def test_success_rate(self):
        """Test success rate calculation."""
        config = BenchmarkConfig()
        results = BenchmarkResults(
            total_requests=1000,
            successful_requests=950,
            failed_requests=50,
            total_time=10.0,
            latencies=[100.0] * 950,
            errors=[],
            metadata={},
            config=config
        )

        assert results.success_rate == 95.0

    def test_latency_stats(self):
        """Test latency statistics calculation."""
        config = BenchmarkConfig()
        latencies = [50.0, 100.0, 150.0, 200.0, 250.0]
        results = BenchmarkResults(
            total_requests=5,
            successful_requests=5,
            failed_requests=0,
            total_time=1.0,
            latencies=latencies,
            errors=[],
            metadata={},
            config=config
        )

        stats = results.get_latency_stats()
        assert stats["count"] == 5
        assert stats["average"] == 150.0
        assert stats["median"] == 150.0
        assert stats["min"] == 50.0
        assert stats["max"] == 250.0

    def test_latency_distribution(self):
        """Test latency distribution calculation."""
        config = BenchmarkConfig()
        latencies = [50.0, 150.0, 300.0, 800.0, 1500.0, 3000.0]
        results = BenchmarkResults(
            total_requests=6,
            successful_requests=6,
            failed_requests=0,
            total_time=1.0,
            latencies=latencies,
            errors=[],
            metadata={},
            config=config
        )

        distribution = results.get_latency_distribution()
        assert len(distribution) == 6

        # Check specific buckets
        labels = [item[0] for item in distribution]
        counts = [item[1] for item in distribution]

        assert "<100ms" in labels
        assert "100-200ms" in labels
        assert "200-500ms" in labels
        assert ">2s" in labels


class TestRedisBenchmarker:
    """Test Redis benchmarker."""

    @patch('redis_benchmarker.benchmarker.get_query_executor')
    @patch('redis.Redis')
    @patch('redis.ConnectionPool')
    def test_run_benchmark_success(self, mock_pool, mock_redis, mock_get_executor):
        """Test successful benchmark run."""
        config = BenchmarkConfig(
            total_requests=10,
            workers=2,
            query_type="mock_executor"
        )

        # Mock executor
        mock_executor_class = Mock(return_value=MockExecutor(config, latency=150.0))
        mock_get_executor.return_value = mock_executor_class

        # Mock Redis
        mock_redis_client = Mock()
        mock_redis_client.ping.return_value = True
        mock_redis_client.close.return_value = None
        mock_redis.return_value = mock_redis_client

        benchmarker = RedisBenchmarker(config)
        results = benchmarker.run_benchmark()

        assert results.total_requests == 10
        assert results.successful_requests == 10
        assert results.failed_requests == 0
        assert len(results.latencies) == 10
        assert all(lat == 150.0 for lat in results.latencies)

    @patch('redis_benchmarker.benchmarker.get_query_executor')
    @patch('redis.Redis')
    @patch('redis.ConnectionPool')
    def test_run_benchmark_with_errors(self, mock_pool, mock_redis, mock_get_executor):
        """Test benchmark run with some errors."""
        config = BenchmarkConfig(
            total_requests=5,
            workers=1,
            query_type="mock_executor"
        )

        # Mock executor that errors sometimes
        executor_instance = MockExecutor(config, should_error=True)
        mock_executor_class = Mock(return_value=executor_instance)
        mock_get_executor.return_value = mock_executor_class

        # Mock Redis
        mock_redis_client = Mock()
        mock_redis_client.ping.return_value = True
        mock_redis_client.close.return_value = None
        mock_redis.return_value = mock_redis_client

        benchmarker = RedisBenchmarker(config)
        results = benchmarker.run_benchmark()

        assert results.total_requests == 5
        assert results.failed_requests == 5
        assert results.successful_requests == 0
        assert len(results.errors) == 5

    def test_format_results(self):
        """Test results formatting."""
        config = BenchmarkConfig(
            total_requests=100,
            workers=4,
            query_type="vector_search"
        )

        results = BenchmarkResults(
            total_requests=100,
            successful_requests=95,
            failed_requests=5,
            total_time=10.0,
            latencies=[100.0] * 95,
            errors=["Error 1", "Error 2"],
            metadata={"query_type": "vector_search"},
            config=config
        )

        benchmarker = RedisBenchmarker(config)
        formatted = benchmarker.format_results(results)

        assert "Redis Query Benchmarker Results" in formatted
        assert "Total Requests: 100" in formatted
        assert "QPS: 9.50" in formatted
        assert "Success Rate: 95.0%" in formatted
        assert "Failed Requests: 5" in formatted

    def test_save_results_json(self):
        """Test saving results to JSON file."""
        config = BenchmarkConfig()
        results = BenchmarkResults(
            total_requests=10,
            successful_requests=10,
            failed_requests=0,
            total_time=1.0,
            latencies=[100.0] * 10,
            errors=[],
            metadata={},
            config=config
        )

        benchmarker = RedisBenchmarker(config)

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_file = f.name

        try:
            benchmarker.save_results(results, temp_file, "json")

            # Verify JSON file
            with open(temp_file, 'r') as f:
                data = json.load(f)

            assert data["summary"]["total_requests"] == 10
            assert data["summary"]["qps"] == 10.0
            assert len(data["raw_latencies"]) == 10

        finally:
            import os
            os.unlink(temp_file)

    def test_save_results_csv(self):
        """Test saving results to CSV file."""
        config = BenchmarkConfig()
        results = BenchmarkResults(
            total_requests=3,
            successful_requests=3,
            failed_requests=0,
            total_time=1.0,
            latencies=[100.0, 200.0, 150.0],
            errors=[],
            metadata={},
            config=config
        )

        benchmarker = RedisBenchmarker(config)

        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            temp_file = f.name

        try:
            benchmarker.save_results(results, temp_file, "csv")

            # Verify CSV file
            with open(temp_file, 'r') as f:
                reader = csv.reader(f)
                rows = list(reader)

            assert rows[0] == ["latency_ms"]
            assert len(rows) == 4  # Header + 3 data rows
            assert float(rows[1][0]) == 100.0
            assert float(rows[2][0]) == 200.0
            assert float(rows[3][0]) == 150.0

        finally:
            import os
            os.unlink(temp_file)

    @patch('redis_benchmarker.benchmarker.get_query_executor')
    def test_unknown_query_type(self, mock_get_executor):
        """Test error handling for unknown query type."""
        mock_get_executor.return_value = None

        config = BenchmarkConfig(query_type="unknown_type")
        benchmarker = RedisBenchmarker(config)

        with pytest.raises(ValueError, match="Unknown query type"):
            benchmarker.run_benchmark()

    @patch('redis_benchmarker.benchmarker.get_query_executor')
    @patch('redis.Redis')
    @patch('redis.ConnectionPool')
    def test_connection_error(self, mock_pool, mock_redis, mock_get_executor):
        """Test connection error handling."""
        config = BenchmarkConfig()

        mock_executor_class = Mock(return_value=MockExecutor(config))
        mock_get_executor.return_value = mock_executor_class

        # Mock Redis connection error
        mock_redis_client = Mock()
        mock_redis_client.ping.side_effect = Exception("Connection failed")
        mock_redis.return_value = mock_redis_client

        benchmarker = RedisBenchmarker(config)

        with pytest.raises(ConnectionError, match="Cannot connect to Redis"):
            benchmarker.run_benchmark()

    @patch('redis_benchmarker.benchmarker.get_query_executor')
    @patch('redis.Redis')
    @patch('redis.ConnectionPool')
    def test_run_benchmark_qps_throttling(self, mock_pool, mock_redis, mock_get_executor):
        """Test QPS throttling in benchmark run."""
        config = BenchmarkConfig(
            total_requests=5,
            workers=2,
            query_type="mock_executor",
            qps=2.0  # 2 queries per second
        )
        mock_executor_class = Mock(return_value=MockExecutor(config, latency=10.0))
        mock_get_executor.return_value = mock_executor_class
        mock_redis_client = Mock()
        mock_redis_client.ping.return_value = True
        mock_redis_client.close.return_value = None
        mock_redis.return_value = mock_redis_client

        benchmarker = RedisBenchmarker(config)
        # Patch time.time and time.sleep to simulate time passing
        times = [100.0]
        def fake_time():
            return times[0]
        def fake_sleep(secs):
            times[0] += secs
        with patch('time.time', fake_time), patch('time.sleep', fake_sleep):
            start = fake_time()
            results = benchmarker.run_benchmark()
            end = fake_time()
        # For 5 requests at 2 QPS, minimum time should be at least 2 seconds
        assert end - start >= 2.0
        assert results.total_requests == 5
        assert results.successful_requests == 5