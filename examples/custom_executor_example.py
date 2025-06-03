#!/usr/bin/env python3
"""
Redis Query Benchmarker - Custom Executor Example

This is a self-contained example showing how to create and use a custom executor
for the Redis Query Benchmarker.

USAGE:
------
Or run the programmatic example:
     PYTHONPATH=. python examples/custom_executor_example.py --run-example


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
from redis_benchmarker.executors import BaseQueryExecutor, register_query_executor
from redis_benchmarker.utils import time_operation
from redis_benchmarker import BenchmarkConfig, RedisBenchmarker


class RedisVLCustomExecutor(BaseQueryExecutor):
    """
    Custom executor using RedisVL for advanced vector search with filters.

    This is a minimal example showing the required methods:
    - prepare(): Set up any resources (indexes, connections, etc.)
    - execute_query(): Execute your custom query and return results
    """

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
        """
        Execute your custom query - called for each benchmark request.

        Must return a dict with:
        - result: Query results
        - latency_ms: Query execution time
        - metadata: Additional info (optional)
        """
        try:
            # Generate random vector and filters for this example
            vector = np.random.random(self.config.vector_dim or 512).astype(np.float32).tolist()
            price_filter = Num("price").between(10, 100)
            category_filter = Tag("category") == "electronics"
            combined_filter = price_filter & category_filter

            # Create and execute vector query with filters
            query = VectorQuery(
                vector=vector,
                vector_field_name="embedding",
                filter_expression=combined_filter,
                return_fields=["title", "category", "price", "vector_distance"],
                num_results=self.config.num_results or 10
            )

            # Time the query execution
            with time_operation() as get_latency_ms:
                result = self.index.query(query)
            latency_ms = get_latency_ms()

            return {
                "result": result,
                "latency_ms": latency_ms,
                "metadata": {
                    "query_type": "filtered_vector_search",
                    "total_results": len(result),
                    "filter_expression": str(combined_filter)
                }
            }
        except Exception as e:
            # Handle errors gracefully
            with time_operation() as get_latency_ms:
                pass  # Just measure the time taken for error handling
            latency_ms = get_latency_ms()

            return {
                "error": str(e),
                "latency_ms": latency_ms,
                "result": None,
                "metadata": {"query_type": "filtered_vector_search"}
            }


# ===== STEP 2: Register Your Custom Executor =====
# Register with a unique name - this makes it available to the benchmarker
register_query_executor("redisvl_custom", RedisVLCustomExecutor)


def main():
    from redis_benchmarker.__main__ import main as cli_command

    if not "--query-type" in sys.argv:
        sys.argv.append("--query-type")
        sys.argv.append("redisvl_custom")

    try:
        cli_command(sys.argv[1:])
    except SystemExit:
        raise


if __name__ == "__main__":
    main()