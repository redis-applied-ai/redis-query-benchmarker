# Redis Query Benchmarker

A flexible benchmarking tool for Redis search queries. Create custom executors to benchmark your specific search patterns with comprehensive performance metrics.

## Quick Start

### 1. Install the Library

Add to your `requirements.txt`:
```
git+https://github.com/redis-applied-ai/redis-query-benchmarker.git
```

Then install:
```bash
pip install -r requirements.txt
```

### 2. Create Your Custom Executor

Create a file `my_executor.py`:

```python
import redis
from redis_benchmarker.executors import BaseQueryExecutor, enable_auto_main
from redis_benchmarker.utils import time_operation
import numpy as np
from redis.commands.search.query import Query

class MySearchExecutor(BaseQueryExecutor):
    def execute_query(self, redis_client: redis.Redis) -> dict:
        """Your custom search logic goes here."""

        # Get pre-generated vector from pool for better performance
        # This avoids expensive vector generation during timing
        query_vector = self.get_vector_from_pool()

        # Time the search operation
        with time_operation() as latency_ms:
            result = redis_client.ft("my_index").search(
                Query("*=>[KNN 10 @embedding $query_vector AS score]")
                .sort_by("score")
                .return_fields("title", "content", "score")
                .dialect(2),
                query_params={"query_vector": query_vector.tobytes()}
            )

        return {
            "result": result,
            "latency_ms": float(latency_ms),
            "metadata": {
                "query_type": "vector_search",
                "total_results": result.total
            }
        }

# This one line enables full CLI functionality
enable_auto_main(__name__)
```

### 3. Run Your Benchmark

```bash
python my_executor.py --total-requests 1000 --workers 16 --index-name my_index
```

That's it! You now have a fully functional benchmark tool with detailed performance metrics.

## Features

- 🚀 **High-performance concurrent benchmarking** with configurable workers
- 📊 **Comprehensive metrics** including latency percentiles, QPS, and distribution analysis
- 🎲 **Built-in data generation** for creating test datasets
- ⚙️ **Flexible configuration** via CLI arguments
- 📈 **Multiple output formats** (console, JSON, CSV)

## Data Generation

Generate test data without needing to check out this repository:

```bash
# Generate sample documents with vectors
python -m redis_benchmarker.data_generator \
  --documents 10000 \
  --vector-dim 1536 \
  --index-name my_index \
  --create-index
```

## Vector Pool for Performance

The benchmarker includes a **vector pool feature** that pre-generates vectors before benchmarking starts. This eliminates the overhead of vector generation during query execution, providing more accurate performance measurements.

### Using the Vector Pool

The vector pool is enabled by default. Simply use `self.get_vector_from_pool()` in your `execute_query` method:

```python
class MyVectorExecutor(BaseQueryExecutor):
    def execute_query(self, redis_client: redis.Redis) -> dict:
        # Get pre-generated vector (much faster than generating during query)
        query_vector = self.get_vector_from_pool()

        with time_operation() as latency_ms:
            result = redis_client.ft("my_index").search(
                Query("*=>[KNN 10 @embedding $query_vector AS score]"),
                query_params={"query_vector": query_vector.tobytes()}
            )

        return {"result": result, "latency_ms": float(latency_ms)}
```

### Customizing Vector Generation

Override `make_single_vector()` to customize how vectors are generated for your use case:

```python
class CustomVectorExecutor(BaseQueryExecutor):
    def make_single_vector(self) -> np.ndarray:
        """Generate custom vectors (e.g., normal distribution, specific patterns)."""
        # Use normal distribution instead of uniform random
        return np.random.randn(self.config.vector_dim).astype(np.float32)

    def prepare(self, redis_client: redis.Redis) -> None:
        # Optionally customize pool size before initialization
        self.set_vector_pool_size(2000)  # Default is 1000
        super().prepare(redis_client)  # This initializes the pool

    def execute_query(self, redis_client: redis.Redis) -> dict:
        query_vector = self.get_vector_from_pool()
        # ... rest of your query logic
```

### Benefits of Vector Pool

- **🚀 Better Performance**: Eliminates vector generation overhead from timing measurements
- **📊 More Accurate Metrics**: Query latency only includes actual search time
- **🎯 Consistent Results**: Same set of vectors used across benchmark runs
- **⚡ Thread-Safe**: Pool efficiently distributes vectors across concurrent workers

## Advanced Usage

### Multiple Executors

Define multiple executors in one file and choose which to run:

```python
class VectorSearchExecutor(BaseQueryExecutor):
    executor_name = "vector_search"  # Custom name

    def execute_query(self, redis_client):
        # Vector search logic
        pass

class HybridSearchExecutor(BaseQueryExecutor):
    executor_name = "hybrid_search"

    def execute_query(self, redis_client):
        # Hybrid search logic
        pass

enable_auto_main(__name__)
```

Run with:
```bash
python my_executor.py --query-type vector_search --total-requests 1000
python my_executor.py --query-type hybrid_search --total-requests 1000
```

### Configuration Options

```bash
python my_executor.py \
    --host localhost \
    --port 6379 \
    --password mypassword \
    --total-requests 1000 \
    --workers 16 \
    --output-format json \
    --output-file results.json
```

## Examples

Check the `examples/` directory for complete working examples:

- **`ultra_simple_executor.py`**: Minimal setup example
- **`standalone_package_example.py`**: Full standalone package structure
- **`custom_executor_example.py`**: RedisVL integration example

Run any example directly:
```bash
git clone https://github.com/redis-applied-ai/redis-query-benchmarker.git
cd redis-query-benchmarker
python examples/ultra_simple_executor.py --total-requests 500
```

## When to Check Out This Repository

You only need to clone this repository if you want to:
- Modify the core benchmarking framework
- Contribute to the project
- Run the provided examples

For normal usage, just install as a dependency and create your custom executors.

## Requirements

- Python 3.8+
- Redis 7.0+ (Redis 8.0+ recommended)
- Your custom dependencies (redis-py, redisvl, etc.)

## Development Setup

Only needed if you want to modify the core library:

```bash
git clone https://github.com/redis-applied-ai/redis-query-benchmarker.git
cd redis-query-benchmarker
pip install -r requirements.txt
pytest  # Run tests
```

## License

[Add your license information here]