# Redis Query Benchmarker

A production-ready, flexible benchmarking tool for Redis search queries with support for both RedisVL and direct redis-py operations.

## Features

- 🚀 **High-performance concurrent benchmarking** with configurable workers and threading
- 🔌 **Pluggable query architecture** - easily implement custom search logic
- 📊 **Comprehensive metrics** including latency percentiles, QPS, and distribution analysis
- 🐳 **Docker support** with Redis container
- 🎲 **Data generation utilities** for creating test datasets with random vectors
- 🧪 **Unit test coverage** for reliability
- ⚙️ **Flexible configuration** via CLI arguments and config files
- 📈 **Multiple output formats** (console, JSON, CSV)

## Quick Start

### Using Docker (Recommended)

1. Start Redis container:
```bash
docker-compose up -d
```

2. Generate sample data:
```bash
python -m redis_benchmarker.data_generator --documents 10000 --vector-dim 1536
```

3. Run benchmark:
```bash
python -m redis_benchmarker --total-requests 1000 --workers 16 --index-name your_index_name
```

### Manual Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Start your Redis instance
3. Follow steps 2-3 above

## Configuration

### Command Line Options

```bash
python -m redis_benchmarker \
    --host localhost \
    --port 6379 \
    --password mypassword \
    --total-requests 1000 \
    --workers 16 \
    --query-type vector_search \
    --index-name your_index_name \
    --output-format json \
    --output-file results.json
```

### Available Query Types

- `vector_search`: RedisVL vector similarity search
- `hybrid_search`: RedisVL hybrid search with filters
- `redis_py`: Direct redis-py FT.SEARCH commands
- **Custom executors**: Register your own query executors (see below)

To see all available query types (including custom ones), run:
```bash
python -c "from redis_benchmarker.executors import list_query_executors; print('\\n'.join(list_query_executors()))"
```

## Custom Query Executors

You can create custom query executors to benchmark any Redis search scenario. This is the **recommended approach** for testing specific query patterns, comparing different approaches, or benchmarking your application's exact search logic.

### Creating a Custom Executor

Create a class that inherits from `BaseQueryExecutor`:

```python
from redis_benchmarker.executors import BaseQueryExecutor, register_query_executor
from redis_benchmarker.utils import time_operation

class MyCustomExecutor(BaseQueryExecutor):
    def execute_query(self, redis_client) -> dict:
        # Your custom search logic here
        with time_operation() as get_latency_ms:
            result = redis_client.ft("my_index").search("@field:value")

        return {
            "result": result,
            "latency_ms": get_latency_ms(),
            "metadata": {"total_results": result.total}
        }

# Register your executor
register_query_executor("my_custom_search", MyCustomExecutor)
```

See `examples/custom_executor_example.py` for a complete working example with RedisVL integration.

### Using Your Custom Executor

**Method 1: Python Script** (Recommended)
```python
from my_custom_executor import MyCustomExecutor  # This registers the executor
from redis_benchmarker import BenchmarkConfig, RedisBenchmarker

config = BenchmarkConfig(
    total_requests=1000,
    workers=16,
    query_type="my_custom_search",
    index_name="my_index"
)

benchmarker = RedisBenchmarker(config)
results = benchmarker.run_benchmark()
benchmarker.print_results(results)
```

**Method 2: Command Line**
```bash
# Run the example directly (includes CLI interface)
python examples/custom_executor_example.py --total-requests 1000 --workers 16
```

### Examples

The `examples/` directory contains several working examples:

- **`custom_executor_example.py`**: Complete custom executor with RedisVL integration
- **`performance_comparison.py`**: Systematic performance testing across different configurations
- **`superlink_executor.py`**: Advanced custom executor example

Run any example directly:
```bash
python examples/custom_executor_example.py --help
python examples/performance_comparison.py
```

### Custom Executor Best Practices

1. **Error Handling**: Always wrap your search logic in try-catch blocks
2. **Timing**: Use `time_operation()` context manager for accurate timing measurements
3. **Metadata**: Include useful metadata in results for analysis
4. **Configuration**: Use `self.config` to access benchmark configuration
5. **Preparation**: Use the `prepare()` method for expensive setup operations
6. **Cleanup**: Use `cleanup()` for any required teardown
7. **Registration**: Always call `register_query_executor()` to make your executor available

### Timing Utilities

The package provides convenient timing utilities:

```python
from redis_benchmarker.utils import time_operation

# Context manager approach (recommended)
with time_operation() as get_latency_ms:
    result = some_operation()
latency = get_latency_ms()
```

## Example Workflows

### Basic Vector Search Benchmark
```bash
# Generate test data
python -m redis_benchmarker.data_generator --documents 10000 --vector-dim 1536

# Run benchmark
python -m redis_benchmarker --query-type vector_search --index-name sample_index --total-requests 1000
```

### Performance Comparison
```bash
# Run systematic performance comparison
python examples/performance_comparison.py
```

### Custom Query Testing
```bash
# Test your custom search logic
python examples/custom_executor_example.py --total-requests 500 --workers 8
```

## Configuration Files

Save and reuse configurations:

```bash
# Save a configuration
python -m redis_benchmarker --save-config my_config.json --query-type vector_search --workers 32

# Load a configuration
python -m redis_benchmarker --config-file my_config.json
```

## Requirements

- Python 3.8+
- Redis 7.0+ (Redis 8.0+ recommended)
- Dependencies listed in `requirements.txt`

## Installation

```bash
git clone <repository-url>
cd redis-query-benchmarker
pip install -r requirements.txt
```

## Development

Run tests:
```bash
pytest
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## License

[Add your license information here]