"""
Redis Query Benchmarker - A production-ready benchmarking tool for Redis search queries.
"""

__version__ = "0.1.0"
__author__ = "Redis Labs"

from .benchmarker import RedisBenchmarker
from .executors import BaseQueryExecutor, register_query_executor, get_query_executor, enable_auto_main
from .config import BenchmarkConfig
from .utils import time_operation, time_function

__all__ = [
    "RedisBenchmarker",
    "BaseQueryExecutor",
    "register_query_executor",
    "get_query_executor",
    "enable_auto_main",  # Deprecated: CLI functionality is now automatic
    "BenchmarkConfig",
    "time_operation",
    "time_function",
]