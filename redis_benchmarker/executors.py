"""Query executor implementations for different Redis search patterns."""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Type
import time
import random
import redis
import numpy as np
from redisvl.index import SearchIndex
from redisvl.query import VectorQuery, FilterQuery

from .utils import time_operation

# Global registry for query executors
_QUERY_EXECUTORS: Dict[str, Type["BaseQueryExecutor"]] = {}


class BaseQueryExecutor(ABC):
    """Base class for query executors."""

    def __init__(self, config: "BenchmarkConfig"):
        self.config = config

    @abstractmethod
    def execute_query(self, redis_client: redis.Redis) -> Dict[str, Any]:
        """
        Execute a single query and return results with timing.

        Args:
            redis_client: Redis connection to use

        Returns:
            Dict containing:
                - result: Query results
                - latency_ms: Query latency in milliseconds
                - metadata: Optional additional metadata
        """
        pass

    def prepare(self, redis_client: redis.Redis) -> None:
        """Optional preparation step before benchmarking starts."""
        pass

    def cleanup(self, redis_client: redis.Redis) -> None:
        """Optional cleanup step after benchmarking ends."""
        pass


class VectorSearchExecutor(BaseQueryExecutor):
    """RedisVL vector search executor."""

    def __init__(self, config: "BenchmarkConfig"):
        super().__init__(config)
        self.index: Optional[SearchIndex] = None
        self.vector_field = config.vector_field or "embedding"

    def prepare(self, redis_client: redis.Redis) -> None:
        """Initialize the search index."""
        if self.config.index_name:
            try:
                self.index = SearchIndex.from_existing(self.config.index_name, redis_client)
            except Exception as e:
                print(f"Warning: Could not load existing index '{self.config.index_name}': {e}")
                self.index = None

    def execute_query(self, redis_client: redis.Redis) -> Dict[str, Any]:
        if not self.index:
            # Create a temporary index reference if not prepared
            if self.config.index_name:
                self.index = SearchIndex.from_existing(self.config.index_name, redis_client)
            else:
                raise ValueError("No index name provided for vector search")

        # Generate random vector
        vector = np.random.random(self.config.vector_dim).astype(np.float32).tolist()

        query = VectorQuery(
            vector=vector,
            vector_field_name=self.vector_field,
            num_results=self.config.num_results,
            return_score=True,
            **self.config.extra_params
        )

        with time_operation() as get_latency_ms:
            result = self.index.query(query)
        latency_ms = get_latency_ms()

        return {
            "result": result,
            "latency_ms": latency_ms,
            "metadata": {"vector_dim": self.config.vector_dim, "num_results": len(result)}
        }


class HybridSearchExecutor(BaseQueryExecutor):
    """RedisVL hybrid search with filters."""

    def __init__(self, config: "BenchmarkConfig"):
        super().__init__(config)
        self.index: Optional[SearchIndex] = None
        self.vector_field = config.vector_field or "embedding"
        self.filter_expression = config.extra_params.get(
            "filter_expression",
            "@price:[0 75]"
        )

    def prepare(self, redis_client: redis.Redis) -> None:
        if self.config.index_name:
            try:
                self.index = SearchIndex.from_existing(self.config.index_name, redis_client)
            except Exception as e:
                print(f"Warning: Could not load existing index '{self.config.index_name}': {e}")

    def execute_query(self, redis_client: redis.Redis) -> Dict[str, Any]:
        if not self.index:
            if self.config.index_name:
                self.index = SearchIndex.from_existing(self.config.index_name, redis_client)
            else:
                raise ValueError("No index name provided for hybrid search")

        vector = np.random.random(self.config.vector_dim).astype(np.float32).tolist()

        query = VectorQuery(
            vector=vector,
            vector_field_name=self.vector_field,
            num_results=self.config.num_results,
            return_score=True,
            filter_expression=self.filter_expression,
            **{k: v for k, v in self.config.extra_params.items() if k != "filter_expression"}
        )

        with time_operation() as get_latency_ms:
            result = self.index.query(query)
        latency_ms = get_latency_ms()

        return {
            "result": result,
            "latency_ms": latency_ms,
            "metadata": {
                "vector_dim": self.config.vector_dim,
                "num_results": len(result),
                "filter_expression": self.filter_expression
            }
        }


class RedisPySearchExecutor(BaseQueryExecutor):
    """Direct redis-py FT.SEARCH executor."""

    def __init__(self, config: "BenchmarkConfig"):
        super().__init__(config)
        self.search_query = config.extra_params.get("search_query", "*")

    def execute_query(self, redis_client: redis.Redis) -> Dict[str, Any]:
        index_name = self.config.index_name or "idx"

        with time_operation() as get_latency_ms:
            result = redis_client.ft(index_name).search(
                self.search_query,
                query_params={"LIMIT": [0, self.config.num_results]}
            )
        latency_ms = get_latency_ms()

        return {
            "result": result,
            "latency_ms": latency_ms,
            "metadata": {
                "total_results": result.total,
                "returned_results": len(result.docs),
                "search_query": self.search_query
            }
        }


# Register built-in executors
def register_query_executor(name: str, executor_class: Type[BaseQueryExecutor]) -> None:
    """Register a query executor."""
    _QUERY_EXECUTORS[name] = executor_class


def get_query_executor(name: str) -> Optional[Type[BaseQueryExecutor]]:
    """Get a registered query executor."""
    return _QUERY_EXECUTORS.get(name)


def list_query_executors() -> List[str]:
    """List all registered query executor names."""
    return list(_QUERY_EXECUTORS.keys())


# Register built-in executors
register_query_executor("vector_search", VectorSearchExecutor)
register_query_executor("hybrid_search", HybridSearchExecutor)
register_query_executor("redis_py", RedisPySearchExecutor)