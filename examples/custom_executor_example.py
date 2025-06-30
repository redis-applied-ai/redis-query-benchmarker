#!/usr/bin/env python3
"""
Redis Query Benchmarker - Custom Executor Example

This is a self-contained example showing how to create and use a custom executor
for the Redis Query Benchmarker.

USAGE:
------
Run directly with CLI options:
     python examples/custom_executor_example.py --total-requests 1000 --workers 16

REQUIREMENTS:
-------------
pip install redis redisvl numpy

Make sure Redis is running locally or update the connection parameters.
"""

import time
import redis
import numpy as np
import sys
from redisvl.index import SearchIndex
from redisvl.query import VectorQuery, FilterQuery
from redisvl.query.filter import Tag, Num
from redis_benchmarker.executors import BaseQueryExecutor, enable_auto_main
from redis_benchmarker.utils import time_operation
from redis_benchmarker import BenchmarkConfig, RedisBenchmarker


class RedisVLCustomExecutor(BaseQueryExecutor):
    """
    Custom executor using RedisVL for advanced vector search with filters.

    This is a minimal example showing the required methods:
    - prepare(): Set up any resources (indexes, connections, etc.)
    - execute_query(): Execute your custom query and return results
    """

    # Optional: specify custom executor name (otherwise uses "redis_v_l_custom")
    executor_name = "redisvl_custom"

    def prepare(self, redis_client: redis.Redis) -> None:
        """Set up RedisVL SearchIndex - called once before benchmarking starts."""
        schema = {
            "index": {
                "name": self.config.index_name or "my_index",
                "prefix": "doc:",
            },
            "fields": [
                {"name": "title", "type": "text"},
                {"name": "category", "type": "tag"},
                {"name": "price", "type": "numeric"},
                {"name": "embedding", "type": "vector", "attrs": {
                    "dims": self.config.vector_dim or 512,
                    "distance_metric": "cosine",
                    "algorithm": "hnsw",
                    "datatype": "float32"
                }}
            ]
        }

        self.index = SearchIndex.from_dict(schema, redis_client=redis_client)
        self.prepared = True

    def execute_query(self, redis_client: redis.Redis) -> dict:
        """Execute a hybrid search query."""
        try:
            # Generate random query vector
            query_vector = np.random.rand(self.config.vector_dim or 512).astype(np.float32)

            # Time the search operation
            with time_operation() as latency_ms:
                result = redis_client.ft(self.config.index_name or "movies").search(
                    Query("(@title:action)=>[KNN 10 @plot_embedding $query_vector AS score]")
                    .sort_by("score")
                    .return_fields("title", "plot", "year", "score")
                    .dialect(2),
                    query_params={"query_vector": query_vector.tobytes()}
                )

            return {
                "result": result,
                "latency_ms": float(latency_ms),
                "metadata": {
                    "query_type": "hybrid_search",
                    "total_results": result.total,
                    "vector_dim": len(query_vector),
                    "text_filter": "action movies"
                }
            }

        except Exception as e:
            # Handle errors gracefully
            with time_operation() as latency_ms:
                pass  # Just measure the error handling time

            return {
                "error": str(e),
                "latency_ms": float(latency_ms),
                "result": None,
                "metadata": {"query_type": "hybrid_search"}
            }


# ===== Enable Auto-Main (replaces registration and main() boilerplate) =====

# This one line replaces all the manual registration and main() function code!
# It automatically discovers RedisVLCustomExecutor, registers it as "redisvl_custom",
# and provides full CLI functionality when run as: python custom_executor_example.py
enable_auto_main(__name__)