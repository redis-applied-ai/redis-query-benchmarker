#!/usr/bin/env python3
"""
Example: Standalone Package using redis-query-benchmarker as dependency

This example shows how you would structure a custom executor as a standalone package
that depends on redis-query-benchmarker from PyPI or GitHub.

STRUCTURE (for a separate repository):
=====================================
my-custom-redis-benchmarks/
├── requirements.txt                    # Dependencies including redis-query-benchmarker
├── my_executor.py                     # This file
├── setup.py                          # Optional: if you want to package this
└── README.md                         # Usage instructions

REQUIREMENTS.txt CONTENTS:
=========================
# For PyPI (when available):
redis-query-benchmarker>=0.1.0

# Or for GitHub dependency (current):
git+https://github.com/redis-applied-ai/redis-query-benchmarker.git

USAGE:
======
python my_executor.py --total-requests 1000 --workers 16 --index-name my_index
"""

import redis
import numpy as np
from redis_benchmarker.executors import BaseQueryExecutor, enable_auto_main
from redis_benchmarker.utils import time_operation
from redis.commands.search.query import Query


class MyAdvancedSearchExecutor(BaseQueryExecutor):
    """
    Custom executor for advanced Redis search patterns.

    This executor demonstrates complex search scenarios specific to your use case.
    """

    # Optional: specify custom executor name (default would be "my_advanced_search")
    executor_name = "advanced_search"

    def prepare(self, redis_client: redis.Redis) -> None:
        """Setup any resources needed for the benchmark."""
        # Example: pre-compute some values, setup connections, etc.
        self.search_templates = [
            "@title:(machine learning)",
            "@category:(AI | ML)",
            "@score:[90 100]",
            "@date:[2024-01-01 2024-12-31]"
        ]
        self.prepared = True

    def execute_query(self, redis_client: redis.Redis) -> dict:
        """
        Execute your custom search logic.

        This might involve:
        - Complex search patterns
        - Multi-step operations
        - Application-specific logic
        - Custom scoring/ranking
        """
        try:
            # Example: rotate through different search patterns
            import random
            search_query = random.choice(self.search_templates)

            # Execute the search with timing
            with time_operation() as latency_ms:
                # Example using FT.SEARCH directly
                result = redis_client.ft(self.config.index_name or "my_index").search(
                    search_query,
                    query_params={
                        "LIMIT": [0, self.config.num_results or 10],
                        "SORTBY": ["score", "DESC"],
                        "RETURN": ["title", "score", "category"]
                    }
                )

            return {
                "result": result,
                "latency_ms": float(latency_ms),
                "metadata": {
                    "query_type": "advanced_search",
                    "search_pattern": search_query,
                    "total_results": result.total,
                    "returned_results": len(result.docs)
                }
            }

        except Exception as e:
            # Graceful error handling
            with time_operation() as latency_ms:
                pass

            return {
                "error": str(e),
                "latency_ms": float(latency_ms),
                "result": None,
                "metadata": {
                    "query_type": "advanced_search",
                    "search_pattern": search_query if 'search_query' in locals() else "unknown"
                }
            }


class MyVectorSimilarityExecutor(BaseQueryExecutor):
    """
    Another custom executor for vector similarity searches.

    This shows how you can have multiple executors in the same module.
    """

    executor_name = "vector_similarity"

    def execute_query(self, redis_client: redis.Redis) -> dict:
        """Execute vector similarity search with custom logic."""
        try:
            # Vector search
            with time_operation() as latency_ms:
                query_vector = np.random.rand(self.config.vector_dim or 128).astype(np.float32)
                # Example: KNN vector search
                result = redis_client.ft(self.config.index_name or "product_search").search(
                    Query("*=>[KNN 10 @embedding $query_vector AS score]")
                    .sort_by("score")
                    .return_fields("id", "title", "description", "score")
                    .dialect(2),
                    query_params={"query_vector": query_vector.tobytes()}
                )

            return {
                "result": result,
                "latency_ms": float(latency_ms),
                "metadata": {
                    "query_type": "vector_search",
                    "total_results": result.total,
                    "vector_dim": len(query_vector)
                }
            }

        except Exception as e:
            return {
                "error": str(e),
                "latency_ms": 0.0,
                "result": None,
                "metadata": {"query_type": "vector_search"}
            }

    def hybrid_search(self, redis_client: redis.Redis) -> dict:
        """Execute a hybrid vector + text search."""
        try:
            with time_operation() as latency_ms:
                query_vector = np.random.rand(self.config.vector_dim or 128).astype(np.float32)
                text_query = self.config.search_query or "electronics"

                # Hybrid search combining vector and text
                # Hybrid search combining vector and text
                result = redis_client.ft(self.config.index_name or "product_search").search(
                    Query(f"(@title:{text_query})=>[KNN 5 @embedding $query_vector AS score]")
                    .sort_by("score")
                    .return_fields("id", "title", "description", "score")
                    .dialect(2),
                    query_params={"query_vector": query_vector.tobytes()}
                )

            return {
                "result": result,
                "latency_ms": float(latency_ms),
                "metadata": {
                    "query_type": "hybrid_search",
                    "total_results": result.total,
                    "text_query": text_query,
                    "vector_dim": len(query_vector)
                }
            }

        except Exception as e:
            return {
                "error": str(e),
                "latency_ms": 0.0,
                "result": None,
                "metadata": {"query_type": "hybrid_search"}
            }

    def filtered_search(self, redis_client: redis.Redis) -> dict:
        """Execute a filtered search with price range."""
        try:
            with time_operation() as latency_ms:
                # Search with price filter
                # Search with price filter
                result = redis_client.ft(self.config.index_name or "product_search").search(
                    Query("@price:[50 200] @category:electronics")
                    .sort_by("price")
                    .return_fields("id", "title", "price", "category")
                )

            return {
                "result": result,
                "latency_ms": float(latency_ms),
                "metadata": {
                    "query_type": "filtered_search",
                    "total_results": result.total,
                    "filter": "price:[50-200] AND category:electronics"
                }
            }

        except Exception as e:
            return {
                "error": str(e),
                "latency_ms": 0.0,
                "result": None,
                "metadata": {"query_type": "filtered_search"}
            }

    def aggregation_search(self, redis_client: redis.Redis) -> dict:
        """Execute an aggregation query."""
        try:
            with time_operation() as latency_ms:
                                # Simple aggregation query example
                result = redis_client.ft(self.config.index_name or "product_search").aggregate(
                    "@*",
                    "GROUPBY", "1", "@category",
                    "REDUCE", "COUNT", "0", "AS", "count",
                    "REDUCE", "AVG", "1", "@price", "AS", "avg_price"
                )

            return {
                "result": result,
                "latency_ms": float(latency_ms),
                "metadata": {
                    "query_type": "aggregation",
                    "total_groups": len(result) if result else 0
                }
            }

        except Exception as e:
            return {
                "error": str(e),
                "latency_ms": 0.0,
                "result": None,
                "metadata": {"query_type": "aggregation"}
            }


# ===== Auto-Main Setup =====

# This automatically discovers both executors and provides CLI functionality
# Users can choose which executor to run with --query-type advanced_search or --query-type vector_similarity
enable_auto_main(__name__)

# Now you can run:
# python my_executor.py --query-type advanced_search --total-requests 1000
# python my_executor.py --query-type vector_similarity --total-requests 500 --vector-dim 1024