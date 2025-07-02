"""Query executor implementations for different Redis search patterns."""

from abc import ABC, abstractmethod, ABCMeta
from typing import Dict, Any, List, Optional, Type
import time
import random
import redis
import numpy as np
import sys
import inspect
import atexit
from redisvl.index import SearchIndex
from redisvl.query import VectorQuery, FilterQuery
from redis.commands.search.query import Query

from .utils import time_operation

# Global registry for query executors
_QUERY_EXECUTORS: Dict[str, Type["BaseQueryExecutor"]] = {}

# Track executors that were defined in __main__ modules
_MAIN_MODULE_EXECUTORS: List[tuple[str, Type["BaseQueryExecutor"]]] = []
_CLI_INVOKED = False


class AutoMainMeta(ABCMeta):
    """
    Metaclass that automatically handles executor registration and CLI functionality.
    Inherits from ABCMeta to work with ABC.

    When a BaseQueryExecutor subclass is defined:
    1. Automatically registers it with a smart name
    2. If defined in a __main__ module, schedules CLI execution
    """

    def __new__(mcs, name, bases, namespace, **kwargs):
        # Create the class normally
        cls = super().__new__(mcs, name, bases, namespace, **kwargs)

        # Only process actual BaseQueryExecutor subclasses (not BaseQueryExecutor itself)
        # Check if this is not the base class and has bases (i.e., it's inheriting from something)
        if (not getattr(cls, '_is_base_class', False) and
            bases and
            name != 'BaseQueryExecutor'):

            # Get the module where this class was defined
            module_name = cls.__module__

            # Auto-register the executor
            executor_name = cls.get_executor_name()
            register_query_executor(executor_name, cls)

            # If this class was defined in a __main__ module, schedule CLI execution
            if module_name == '__main__':
                _MAIN_MODULE_EXECUTORS.append((executor_name, cls))
                # Set up atexit handler to run CLI when the script ends
                if not _CLI_INVOKED:
                    atexit.register(_auto_invoke_cli)

        return cls


def _auto_invoke_cli():
    """
    Automatically invoke CLI if executors were defined in __main__ module.
    Called via atexit when the module execution is complete.
    """
    global _CLI_INVOKED

    # Only invoke once
    if _CLI_INVOKED or not _MAIN_MODULE_EXECUTORS:
        return

    _CLI_INVOKED = True

    # Determine which executor to use as default
    executor_names = [name for name, _ in _MAIN_MODULE_EXECUTORS]

    if len(executor_names) == 1:
        # Single executor - auto-add as default if not specified
        default_name = executor_names[0]
        if "--query-type" not in sys.argv:
            sys.argv.extend(["--query-type", default_name])
    else:
        # Multiple executors - require user to specify
        if "--query-type" not in sys.argv:
            print(f"Multiple executors found: {', '.join(executor_names)}")
            print("Please specify which one to use with --query-type")
            sys.exit(1)

    # Import and run the CLI
    try:
        # Try relative import first (when running from package)
        try:
            from .__main__ import main as cli_main
        except ImportError:
            # Fall back to absolute import (when running as script)
            from redis_benchmarker.__main__ import main as cli_main
        cli_main()
    except SystemExit:
        # Click uses SystemExit for --help and error conditions
        raise
    except ImportError as e:
        print(f"Error: Could not import CLI module: {e}", file=sys.stderr)
        sys.exit(1)


class BaseQueryExecutor(ABC, metaclass=AutoMainMeta):
    """Base class for query executors with automatic registration and CLI functionality."""

    # Mark this as the base class to avoid self-registration
    _is_base_class = True

    def __init__(self, config: "BenchmarkConfig"):
        self.config = config
        self._vector_pool: List[np.ndarray] = []
        self._vector_pool_index = 0
        self._vector_pool_size = 1000  # Default pool size

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
        # Pre-generate vector pool for performance
        self._initialize_vector_pool()

    def _initialize_vector_pool(self) -> None:
        """Initialize the vector pool with pre-generated vectors."""
        if not self._vector_pool:  # Only initialize if empty
            print(f"Pre-generating {self._vector_pool_size} vectors of dimension {self.config.vector_dim} for performance...")
            self._vector_pool = []
            for _ in range(self._vector_pool_size):
                vector = self.make_single_vector()
                self._vector_pool.append(vector)
            print(f"Vector pool initialized with {len(self._vector_pool)} vectors")

    def make_single_vector(self) -> np.ndarray:
        """
        Generate a single random vector. Override this method to customize vector generation.

        Returns:
            np.ndarray: Random vector of shape (vector_dim,) as float32
        """
        return np.random.random(self.config.vector_dim).astype(np.float32)

    def get_vector_from_pool(self) -> np.ndarray:
        """
        Get a pre-generated vector from the pool for use in queries.

        This method is thread-safe and cycles through the pool.
        The returned vector should be used immediately (e.g., converted to list/bytes)
        and not modified to ensure thread safety.

        Returns:
            np.ndarray: A vector from the pre-generated pool (direct reference, no copy)
        """
        if not self._vector_pool:
            # Fallback if pool not initialized
            return self.make_single_vector()

        # Use modulo to cycle through the pool (thread-safe for reads)
        vector = self._vector_pool[self._vector_pool_index % len(self._vector_pool)]
        self._vector_pool_index = (self._vector_pool_index + 1) % len(self._vector_pool)
        return vector  # Return direct reference for maximum performance

    def set_vector_pool_size(self, size: int) -> None:
        """
        Set the vector pool size. Call this before prepare() to take effect.

        Args:
            size: Number of vectors to pre-generate
        """
        if size < 1:
            raise ValueError("Vector pool size must be at least 1")
        self._vector_pool_size = size
        # Clear existing pool so it gets regenerated with new size
        self._vector_pool = []

    def cleanup(self, redis_client: redis.Redis) -> None:
        """Optional cleanup step after benchmarking ends."""
        pass

    @classmethod
    def get_executor_name(cls) -> str:
        """
        Get the executor name for registration.

        Override the 'executor_name' class attribute to specify a custom name,
        otherwise uses the class name converted to snake_case.
        """
        if hasattr(cls, 'executor_name') and cls.executor_name:
            return cls.executor_name

        # Convert class name to snake_case
        name = cls.__name__
        if name.endswith('Executor'):
            name = name[:-8]  # Remove 'Executor' suffix

        # Convert CamelCase to snake_case
        result = []
        for i, char in enumerate(name):
            if char.isupper() and i > 0:
                result.append('_')
            result.append(char.lower())

        return ''.join(result)


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

        # Get pre-generated vector from pool
        vector = self.get_vector_from_pool().tolist()

        query = VectorQuery(
            vector=vector,
            vector_field_name=self.vector_field,
            num_results=self.config.num_results,
            return_score=True,
            **self.config.extra_params
        )

        with time_operation() as latency_ms:
            result = self.index.query(query)

        return {
            "result": result,
            "latency_ms": float(latency_ms),
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

        vector = self.get_vector_from_pool().tolist()

        query = VectorQuery(
            vector=vector,
            vector_field_name=self.vector_field,
            num_results=self.config.num_results,
            return_score=True,
            filter_expression=self.filter_expression,
            **{k: v for k, v in self.config.extra_params.items() if k != "filter_expression"}
        )

        with time_operation() as latency_ms:
            result = self.index.query(query)

        return {
            "result": result,
            "latency_ms": float(latency_ms),
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

        with time_operation() as latency_ms:
            result = redis_client.ft(index_name).search(
                self.search_query,
                query_params={"LIMIT": [0, self.config.num_results]}
            )

        return {
            "result": result,
            "latency_ms": float(latency_ms),
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


def _discover_executors_in_module(module_name: str) -> List[Type[BaseQueryExecutor]]:
    """
    Discover all BaseQueryExecutor subclasses in the given module.

    Args:
        module_name: The name of the module to inspect (usually __name__)

    Returns:
        List of discovered executor classes
    """
    if module_name not in sys.modules:
        return []

    module = sys.modules[module_name]
    executors = []

    for name, obj in inspect.getmembers(module, inspect.isclass):
        if (obj != BaseQueryExecutor and
            issubclass(obj, BaseQueryExecutor) and
            obj.__module__ == module_name):
            executors.append(obj)

    return executors


def enable_auto_main(module_name: str, default_executor_name: Optional[str] = None) -> None:
    """
    Enable automatic CLI functionality for custom executors.

    NOTE: This function is now DEPRECATED. Simply inheriting from BaseQueryExecutor
    automatically provides CLI functionality. This function is kept for backward compatibility.

    Args:
        module_name: The module name (pass __name__)
        default_executor_name: Optional specific executor name to use as default
    """
    # Only activate if this module is being run as main
    if module_name != '__main__':
        return

    # Discover and register executors in the module
    executors = _discover_executors_in_module(module_name)

    if not executors:
        print("Error: No BaseQueryExecutor subclasses found in this module", file=sys.stderr)
        sys.exit(1)

    # Register all discovered executors
    registered_names = []
    for executor_class in executors:
        executor_name = executor_class.get_executor_name()
        register_query_executor(executor_name, executor_class)
        registered_names.append(executor_name)

    # Determine which executor to use as default
    if default_executor_name and default_executor_name in registered_names:
        default_name = default_executor_name
    elif len(registered_names) == 1:
        default_name = registered_names[0]
    else:
        # Multiple executors found, let user choose or specify
        if "--query-type" not in sys.argv:
            print(f"Multiple executors found: {', '.join(registered_names)}")
            print("Please specify which one to use with --query-type")
            sys.exit(1)
        default_name = None

    # Add default query type if not specified
    if default_name and "--query-type" not in sys.argv:
        sys.argv.extend(["--query-type", default_name])

    # Import and run the CLI
    try:
        # Try relative import first (when running from package)
        try:
            from .__main__ import main as cli_main
        except ImportError:
            # Fall back to absolute import (when running as script)
            from redis_benchmarker.__main__ import main as cli_main
        cli_main()
    except SystemExit:
        # Click uses SystemExit for --help and error conditions
        raise
    except ImportError as e:
        print(f"Error: Could not import CLI module: {e}", file=sys.stderr)
        sys.exit(1)


# Register built-in executors
register_query_executor("vector_search", VectorSearchExecutor)
register_query_executor("hybrid_search", HybridSearchExecutor)
register_query_executor("redis_py", RedisPySearchExecutor)