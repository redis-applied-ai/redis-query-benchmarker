"""Main Redis benchmarker implementation."""

import time
import json
import csv
import threading
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import redis
import numpy as np
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, TaskID, BarColumn, TextColumn, TimeRemainingColumn, SpinnerColumn

from .config import BenchmarkConfig
from .executors import BaseQueryExecutor, get_query_executor


@dataclass
class BenchmarkResults:
    """Container for benchmark results."""

    total_requests: int
    successful_requests: int
    failed_requests: int
    total_time: float
    latencies: List[float]
    errors: List[str]
    metadata: Dict[str, Any]
    config: BenchmarkConfig
    result_counts: List[int]

    @property
    def qps(self) -> float:
        """Calculate queries per second."""
        return self.successful_requests / self.total_time if self.total_time > 0 else 0.0

    @property
    def success_rate(self) -> float:
        """Calculate success rate percentage."""
        return (self.successful_requests / self.total_requests) * 100 if self.total_requests > 0 else 0.0

    @property
    def average_result_count(self) -> float:
        """Calculate average number of results per query."""
        return np.mean(self.result_counts) if self.result_counts else 0.0

    @property
    def average_normalized_latency(self) -> float:
        """Calculate average normalized latency (latency per result)."""
        if not self.latencies or not self.result_counts or len(self.latencies) != len(self.result_counts):
            return 0.0

        # Use numpy for vectorized operations - much faster
        latencies_array = np.array(self.latencies)
        counts_array = np.array(self.result_counts)

        # Only include queries that returned results (count > 0)
        valid_mask = counts_array > 0
        if not np.any(valid_mask):
            return 0.0

        return float(np.mean(latencies_array[valid_mask] / counts_array[valid_mask]))

    def get_latency_stats(self) -> Dict[str, float]:
        """Calculate latency statistics."""
        if not self.latencies:
            return {}

        latencies_array = np.array(self.latencies)
        return {
            "count": len(self.latencies),
            "average": float(np.mean(latencies_array)),
            "median": float(np.median(latencies_array)),
            "min": float(np.min(latencies_array)),
            "max": float(np.max(latencies_array)),
            "std_dev": float(np.std(latencies_array)),
            "p90": float(np.percentile(latencies_array, 90)),
            "p95": float(np.percentile(latencies_array, 95)),
            "p99": float(np.percentile(latencies_array, 99)),
        }

    def get_latency_distribution(self) -> List[Tuple[str, int, float]]:
        """Get latency distribution in buckets."""
        if not self.latencies:
            return []

        bins = [0, 100, 200, 500, 1000, 2000, float("inf")]
        labels = ["<100ms", "100-200ms", "200-500ms", "500ms-1s", "1-2s", ">2s"]

        # Use numpy histogram for O(n) performance instead of O(n*bins)
        latencies_array = np.array(self.latencies)
        counts, _ = np.histogram(latencies_array, bins=bins)

        total = len(self.latencies)
        distribution = []
        for count, label in zip(counts, labels):
            percentage = (count / total) * 100
            distribution.append((label, int(count), percentage))

        return distribution


class RedisBenchmarker:
    """Main Redis benchmarker class."""

    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.console = Console()
        self._connection_pool: Optional[redis.ConnectionPool] = None

    def _create_connection_pool(self) -> redis.ConnectionPool:
        """Create Redis connection pool."""
        if self._connection_pool is None:
            pool_kwargs = {
                "host": self.config.redis.host,
                "port": self.config.redis.port,
                "db": self.config.redis.db,
                "socket_timeout": self.config.redis.socket_timeout,
                "socket_connect_timeout": self.config.redis.socket_connect_timeout,
                "max_connections": self.config.max_connections,
            }

            if self.config.redis.password:
                pool_kwargs["password"] = self.config.redis.password
            if self.config.redis.username:
                pool_kwargs["username"] = self.config.redis.username

            # Configure TLS/SSL if enabled - use SSLConnection class for TLS
            if self.config.redis.tls:
                pool_kwargs.update({
                    "connection_class": redis.connection.SSLConnection,
                    "ssl_cert_reqs": "none" if self.config.redis.tls_insecure else "required",
                    "ssl_check_hostname": not self.config.redis.tls_insecure,
                })
            else:
                pool_kwargs["connection_class"] = redis.connection.Connection

            self._connection_pool = redis.ConnectionPool(**pool_kwargs)

        return self._connection_pool

    def _get_redis_client(self) -> redis.Redis:
        """Get Redis client using the connection pool."""
        return redis.Redis(connection_pool=self._create_connection_pool())

    def _execute_single_query(self, executor: BaseQueryExecutor, connection_pool: redis.ConnectionPool) -> Dict[str, Any]:
        """Execute a single query with error handling."""
        redis_client = redis.Redis(connection_pool=connection_pool)
        try:
            return executor.execute_query(redis_client)
        except Exception as e:
            return {
                "error": str(e),
                "latency_ms": 0,
                "result": None,
                "metadata": {}
            }
        # Note: Don't close the client when using connection pool - connections are reused

    def _extract_result_count(self, query_result: Dict[str, Any]) -> int:
        """Extract result count from query result metadata."""
        metadata = query_result.get("metadata", {})

        # Check for different result count fields based on executor type
        if "num_results" in metadata:
            return metadata["num_results"]
        elif "returned_results" in metadata:
            return metadata["returned_results"]

        # Fallback: try to get length from result directly
        result = query_result.get("result")
        if result is not None:
            try:
                if hasattr(result, '__len__'):
                    return len(result)
                elif hasattr(result, 'docs') and hasattr(result.docs, '__len__'):
                    return len(result.docs)
            except (TypeError, AttributeError):
                pass

        return 0

    def _run_warmup(self, executor: BaseQueryExecutor) -> None:
        """Run warmup queries to prepare connections and caches."""
        if self.config.warmup_requests <= 0:
            return

        self.console.print(f"Running {self.config.warmup_requests} warmup queries...")

        # Create connection pool for warmup
        connection_pool = self._create_connection_pool()

        with ThreadPoolExecutor(max_workers=min(self.config.workers, 8)) as executor_pool:
            futures = [
                executor_pool.submit(self._execute_single_query, executor, connection_pool)
                for _ in range(self.config.warmup_requests)
            ]

            for future in as_completed(futures):
                try:
                    future.result()  # Just consume the result
                except Exception:
                    pass  # Ignore warmup errors

    def _calculate_metrics(self, start_time: float, successful_count: int, total_latency: float, total_result_count: int) -> Dict[str, float]:
        """Calculate current metrics efficiently."""
        elapsed_time = time.time() - start_time

        if successful_count > 0:
            avg_latency = total_latency / successful_count
            avg_results = total_result_count / successful_count
        else:
            avg_latency = 0.0
            avg_results = 0.0

        if elapsed_time > 0:
            current_qps = successful_count / elapsed_time
        else:
            current_qps = 0.0

        norm_latency = avg_latency / avg_results if avg_results > 0 else 0.0

        return {
            "avg_latency": avg_latency,
            "avg_results": avg_results,
            "current_qps": current_qps,
            "norm_latency": norm_latency
        }

    def run_benchmark(self) -> BenchmarkResults:
        """Run the main benchmark."""
        # Get query executor
        executor_class = get_query_executor(self.config.query_type)
        if not executor_class:
            raise ValueError(f"Unknown query type: {self.config.query_type}")

        executor = executor_class(self.config)

        # Test connection
        try:
            test_client = self._get_redis_client()
            test_client.ping()
            test_client.close()
        except Exception as e:
            raise ConnectionError(f"Cannot connect to Redis: {e}")

        # Prepare executor
        prep_client = self._get_redis_client()
        try:
            executor.prepare(prep_client)
        finally:
            prep_client.close()

        # Run warmup
        self._run_warmup(executor)

        # Run main benchmark
        self.console.print(f"Starting benchmark with {self.config.total_requests} requests using {self.config.workers} workers...")

        # Preallocate lists for better memory efficiency - use deque for O(1) appends
        from collections import deque
        results = deque()
        latencies = deque()
        errors = deque()
        result_counts = deque()

        start_time = time.time()

        # Variables for efficient running average calculation
        total_latency = 0.0
        total_result_count = 0
        successful_count = 0

        # Custom progress bar with real-time metrics
        progress_columns = [
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TextColumn("•"),
            TextColumn("[bold blue]{task.completed}/{task.total}"),
            TextColumn("•"),
            TextColumn("[bold green]Avg: {task.fields[avg_latency]:.1f}ms"),
            TextColumn("•"),
            TextColumn("[bold cyan]QPS: {task.fields[current_qps]:.1f}"),
        ]

        if self.config.show_expanded_metrics:
            progress_columns.extend([
                TextColumn("•"),
                TextColumn("[bold yellow]Avg Results: {task.fields[avg_results]:.1f}"),
                TextColumn("•"),
                TextColumn("[bold magenta]Norm Lat: {task.fields[norm_latency]:.2f}ms/result"),
            ])

        progress_columns.append(TimeRemainingColumn())

        interrupted = False
        interrupt_time = None
        # Create connection pool once for all workers
        connection_pool = self._create_connection_pool()

        with Progress(
                *progress_columns,
                console=self.console,
                expand=True
            ) as progress:
                task_fields = {
                    "avg_latency": 0.0,
                    "current_qps": 0.0
                }

                if self.config.show_expanded_metrics:
                    task_fields.update({
                        "avg_results": 0.0,
                        "norm_latency": 0.0
                    })

                task = progress.add_task(
                    "Benchmarking...",
                    total=self.config.total_requests,
                    **task_fields
                )

                # Use ThreadPoolExecutor with simplified future management
                with ThreadPoolExecutor(max_workers=self.config.workers) as thread_executor:
                    try:
                        # Submit all futures at once (with QPS limiting if needed)
                        futures = []
                        if self.config.qps:
                            # QPS limited submission
                            next_submit_time = time.time()
                            for i in range(self.config.total_requests):
                                now = time.time()
                                if now < next_submit_time:
                                    time.sleep(next_submit_time - now)
                                next_submit_time = max(next_submit_time + 1.0 / self.config.qps, time.time())

                                future = thread_executor.submit(self._execute_single_query, executor, connection_pool)
                                futures.append(future)
                        else:
                            # No QPS limit - submit all at once
                            futures = [
                                thread_executor.submit(self._execute_single_query, executor, connection_pool)
                                for _ in range(self.config.total_requests)
                            ]

                        # Process completions using as_completed for efficiency
                        completed_count = 0
                        batch_size = max(1, self.config.workers // 4)  # Update progress in batches
                        batch_count = 0

                        for future in as_completed(futures):
                            try:
                                query_result = future.result(timeout=self.config.timeout)

                                if "error" in query_result:
                                    errors.append(query_result["error"])
                                else:
                                    results.append(query_result["result"])
                                    latencies.append(query_result["latency_ms"])

                                    # Extract result count from metadata
                                    result_count = self._extract_result_count(query_result)
                                    result_counts.append(result_count)

                                    # Update running averages efficiently
                                    total_latency += query_result["latency_ms"]
                                    total_result_count += result_count
                                    successful_count += 1

                                completed_count += 1
                                batch_count += 1

                                # Update progress in batches to reduce overhead
                                if batch_count >= batch_size or completed_count == self.config.total_requests:
                                    metrics = self._calculate_metrics(start_time, successful_count, total_latency, total_result_count)

                                    # Update progress bar with new metrics
                                    progress.update(task, completed=completed_count)
                                    update_fields = {"avg_latency": metrics["avg_latency"], "current_qps": metrics["current_qps"]}

                                    if self.config.show_expanded_metrics:
                                        update_fields.update({
                                            "avg_results": metrics["avg_results"],
                                            "norm_latency": metrics["norm_latency"]
                                        })

                                    progress.update(task, **update_fields)
                                    batch_count = 0

                            except Exception as e:
                                errors.append(str(e))
                                completed_count += 1
                                batch_count += 1

                                # Update progress even on error (batched)
                                if batch_count >= batch_size or completed_count == self.config.total_requests:
                                    metrics = self._calculate_metrics(start_time, successful_count, total_latency, total_result_count)

                                    progress.update(task, completed=completed_count)
                                    update_fields = {"avg_latency": metrics["avg_latency"], "current_qps": metrics["current_qps"]}

                                    if self.config.show_expanded_metrics:
                                        update_fields.update({
                                            "avg_results": metrics["avg_results"],
                                            "norm_latency": metrics["norm_latency"]
                                        })

                                    progress.update(task, **update_fields)
                                    batch_count = 0

                    except KeyboardInterrupt:
                        interrupted = True
                        interrupt_time = time.time()

                        # Print message immediately
                        self.console.print("\n[yellow]Benchmark interrupted by user. Collecting partial results...[/yellow]")

                        # Cancel any pending futures
                        for future in futures:
                            if not future.done():
                                future.cancel()

                        # Stop progress display
                        try:
                            progress.stop()
                        except:
                            pass

        # Calculate end time - use interrupt time if interrupted, otherwise current time
        end_time = interrupt_time if interrupted else time.time()
        total_time = end_time - start_time

        # Cleanup
        cleanup_client = self._get_redis_client()
        try:
            executor.cleanup(cleanup_client)
        finally:
            cleanup_client.close()

        return BenchmarkResults(
            total_requests=self.config.total_requests,
            successful_requests=len(latencies),
            failed_requests=len(errors),
            total_time=total_time,
            latencies=list(latencies),  # Convert deque to list for final results
            errors=list(errors),
            metadata={
                "query_type": self.config.query_type,
                "workers": self.config.workers,
                "redis_host": f"{self.config.redis.host}:{self.config.redis.port}",
                "interrupted": interrupted,
                "completed_requests": len(latencies) + len(errors),
            },
            config=self.config,
            result_counts=list(result_counts)
        )

    def format_results(self, results: BenchmarkResults) -> str:
        """Format results for console output."""
        output = []
        output.append("Redis Query Benchmarker Results")
        output.append("=" * 40)
        output.append("")

        # Show interruption status if applicable
        if results.metadata.get("interrupted", False):
            completed = results.metadata.get("completed_requests", results.successful_requests + results.failed_requests)
            output.append(f"[INTERRUPTED] Benchmark was stopped early (completed {completed}/{results.total_requests} requests)")
            output.append("")

        # Configuration
        output.append("Configuration:")
        output.append(f"  Host: {results.config.redis.host}:{results.config.redis.port}")
        output.append(f"  Workers: {results.config.workers}")
        output.append(f"  Total Requests: {results.total_requests}")
        output.append(f"  Query Type: {results.config.query_type}")
        output.append("")

        # Results summary
        output.append("Results:")
        output.append(f"  Sent {results.successful_requests} queries in {results.total_time:.2f} seconds")
        output.append(f"  Average QPS: {results.qps:.2f}")
        output.append(f"  Success Rate: {results.success_rate:.1f}%")

        if results.result_counts:
            output.append(f"  Average Results per Query: {results.average_result_count:.2f}")
            output.append(f"  Average Normalized Latency: {results.average_normalized_latency:.2f}ms/result")

        if results.failed_requests > 0:
            output.append(f"  Failed Requests: {results.failed_requests}")

        # Latency statistics
        if results.latencies:
            stats = results.get_latency_stats()
            output.append("")
            output.append("Latency Statistics (ms):")
            output.append(f"  Count:      {stats['count']:,}")
            output.append(f"  Average:    {stats['average']:.2f}")
            output.append(f"  Median:     {stats['median']:.2f}")
            output.append(f"  Min:        {stats['min']:.2f}")
            output.append(f"  Max:        {stats['max']:.2f}")
            output.append(f"  Std Dev:    {stats['std_dev']:.2f}")
            output.append(f"  90th pct:   {stats['p90']:.2f}")
            output.append(f"  95th pct:   {stats['p95']:.2f}")
            output.append(f"  99th pct:   {stats['p99']:.2f}")

            # Distribution
            output.append("")
            output.append("Latency Distribution:")
            for label, count, percentage in results.get_latency_distribution():
                output.append(f"  {label:>10}: {count:4d} ({percentage:5.1f}%)")

        # Errors
        if results.errors:
            output.append("")
            output.append("Sample Errors:")
            for error in results.errors[:5]:  # Show first 5 errors
                output.append(f"  - {error}")
            if len(results.errors) > 5:
                output.append(f"  ... and {len(results.errors) - 5} more")

        return "\n".join(output)

    def save_results(self, results: BenchmarkResults, output_file: str, output_format: str) -> None:
        """Save results to file."""
        if output_format == "json":
            data = {
                "config": results.config.to_dict(),
                "summary": {
                    "total_requests": results.total_requests,
                    "successful_requests": results.successful_requests,
                    "failed_requests": results.failed_requests,
                    "total_time": results.total_time,
                    "qps": results.qps,
                    "success_rate": results.success_rate,
                    "average_result_count": results.average_result_count,
                    "average_normalized_latency": results.average_normalized_latency,
                },
                "latency_stats": results.get_latency_stats(),
                "latency_distribution": [
                    {"bucket": label, "count": count, "percentage": percentage}
                    for label, count, percentage in results.get_latency_distribution()
                ],
                "raw_latencies": results.latencies,
                "raw_result_counts": results.result_counts,
                "errors": results.errors,
                "metadata": results.metadata,
            }

            with open(output_file, 'w') as f:
                json.dump(data, f, indent=2)

        elif output_format == "csv":
            with open(output_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["latency_ms"])
                for latency in results.latencies:
                    writer.writerow([latency])

        self.console.print(f"Results saved to: {output_file}")

    def print_results(self, results: BenchmarkResults) -> None:
        """Print formatted results to console."""
        self.console.print(self.format_results(results))