"""Tests for executor module."""

import pytest
from unittest.mock import Mock, MagicMock
import numpy as np

from redis_benchmarker.config import BenchmarkConfig
from redis_benchmarker.executors import (
    BaseQueryExecutor, VectorSearchExecutor, HybridSearchExecutor,
    RedisPySearchExecutor,
    register_query_executor, get_query_executor, list_query_executors
)


class MockRedisClient:
    """Mock Redis client for testing."""

    def __init__(self):
        self.ping = Mock(return_value=True)
        self.close = Mock()
        self.ft = Mock()

        # Mock search index
        self.search_result = Mock()
        self.search_result.total = 100
        self.search_result.docs = [Mock() for _ in range(10)]
        self.ft.return_value.search.return_value = self.search_result


class TestBaseQueryExecutor:
    """Test base executor functionality."""

    def test_abstract_class(self):
        """Test that BaseQueryExecutor is abstract."""
        with pytest.raises(TypeError):
            BaseQueryExecutor(BenchmarkConfig())


class TestVectorSearchExecutor:
    """Test vector search executor."""

    def test_initialization(self):
        """Test executor initialization."""
        config = BenchmarkConfig(vector_dim=512, vector_field="test_embedding")
        executor = VectorSearchExecutor(config)

        assert executor.config == config
        assert executor.vector_field == "test_embedding"
        assert executor.index is None

    def test_execute_query_no_index(self):
        """Test query execution without prepared index."""
        config = BenchmarkConfig(
            index_name="test_index",
            vector_dim=128,
            num_results=5
        )
        executor = VectorSearchExecutor(config)

        redis_client = MockRedisClient()

        # Mock SearchIndex.from_existing to raise an exception and test fallback
        with pytest.raises(ValueError, match="No index name provided"):
            config_no_index = BenchmarkConfig(vector_dim=128)
            executor_no_index = VectorSearchExecutor(config_no_index)
            executor_no_index.execute_query(redis_client)


class TestHybridSearchExecutor:
    """Test hybrid search executor."""

    def test_initialization(self):
        """Test executor initialization."""
        config = BenchmarkConfig(
            extra_params={"filter_expression": "@category:{electronics}"}
        )
        executor = HybridSearchExecutor(config)

        assert executor.filter_expression == "@category:{electronics}"

    def test_default_filter(self):
        """Test default filter expression."""
        config = BenchmarkConfig()
        executor = HybridSearchExecutor(config)

        assert executor.filter_expression == "@price:[0 75]"


class TestRedisPySearchExecutor:
    """Test redis-py search executor."""

    def test_initialization(self):
        """Test executor initialization."""
        config = BenchmarkConfig(
            extra_params={"search_query": "@title:test"}
        )
        executor = RedisPySearchExecutor(config)

        assert executor.search_query == "@title:test"

    def test_default_query(self):
        """Test default search query."""
        config = BenchmarkConfig()
        executor = RedisPySearchExecutor(config)

        assert executor.search_query == "*"

    def test_execute_query(self):
        """Test query execution."""
        config = BenchmarkConfig(
            index_name="test_index",
            num_results=5,
            extra_params={"search_query": "@title:test"}
        )
        executor = RedisPySearchExecutor(config)

        redis_client = MockRedisClient()
        result = executor.execute_query(redis_client)

        # Verify ft().search was called
        redis_client.ft.assert_called_with("test_index")
        redis_client.ft.return_value.search.assert_called_once()

        # Check result structure
        assert "result" in result
        assert "latency_ms" in result
        assert "metadata" in result
        assert result["metadata"]["search_query"] == "@title:test"


class TestExecutorRegistry:
    """Test executor registration system."""

    def test_register_and_get_executor(self):
        """Test registering and retrieving executors."""

        class TestExecutor(BaseQueryExecutor):
            def execute_query(self, redis_client):
                return {"result": "test", "latency_ms": 1.0}

        # Register new executor
        register_query_executor("test_executor", TestExecutor)

        # Retrieve executor
        executor_class = get_query_executor("test_executor")
        assert executor_class == TestExecutor

        # Check it's in the list
        assert "test_executor" in list_query_executors()

    def test_get_nonexistent_executor(self):
        """Test getting non-existent executor returns None."""
        result = get_query_executor("nonexistent_executor")
        assert result is None

    def test_builtin_executors(self):
        """Test that built-in executors are registered."""
        executors = list_query_executors()

        assert "vector_search" in executors
        assert "hybrid_search" in executors
        assert "redis_py" in executors