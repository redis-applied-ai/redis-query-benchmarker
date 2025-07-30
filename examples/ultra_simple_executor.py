#!/usr/bin/env python3
"""
Ultra Simple Custom Executor - Minimal Boilerplate!

This example shows the simplest possible custom executor.
Just one line of setup code needed!

USAGE:
------
python ultra_simple_executor.py --total-requests 1000 --workers 16

# For better performance with many workers, pre-warm connections:
# python ultra_simple_executor.py --total-requests 1000 --workers 16 --pre-warm 16

That's it! Minimal setup, maximum functionality.
"""

import redis
import time
from redis_benchmarker.executors import BaseQueryExecutor, enable_auto_main
from redis_benchmarker.utils import time_operation


class MySimpleSearchExecutor(BaseQueryExecutor):
    """
    A simple custom executor that demonstrates the minimal-boilerplate approach.

    This class is automatically:
    - Registered as "my_simple_search" (class name → snake_case)
    - Available via CLI when this script is run
    - Ready to use with all command-line options
    """

    def execute_query(self, redis_client: redis.Redis) -> dict:
        """Your custom search logic goes here."""

        try:
            # Example: Simple Redis search
            with time_operation() as latency_ms:
                # Replace this with your actual search logic
                result = redis_client.ft(self.config.index_name or "my_index").search(
                    "@title:python",
                    query_params={"LIMIT": [0, self.config.num_results or 10]}
                )

            return {
                "result": result,
                "latency_ms": float(latency_ms),
                "metadata": {
                    "query_type": "simple_search",
                    "total_results": result.total,
                    "search_term": "python"
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
                "metadata": {"query_type": "simple_search"}
            }


# This one line enables full CLI functionality!
enable_auto_main(__name__)

# That's it! Just one line of setup.
#
# This automatically:
# - Registers the class as "my_simple_search"
# - Provides full CLI functionality
# - Makes it runnable as: python ultra_simple_executor.py
#
# Want a custom name? Add this to your class:
# executor_name = "my_custom_name"
#
# Want multiple executors? Define multiple classes and choose with --query-type