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

    @property
    def qps(self) -> float:
        """Calculate queries per second."""
        return self.successful_requests / self.total_time if self.total_time > 0 else 0.0

    @property
    def success_rate(self) -> float:
        """Calculate success rate percentage."""
        return (self.successful_requests / self.total_requests) * 100 if self.total_requests > 0 else 0.0

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

        distribution = []
        for i, (bin_max, label) in enumerate(zip(bins[1:], labels)):
            bin_min = bins[i]
            count = sum(1 for lat in self.latencies if bin_min <= lat < bin_max)
            percentage = (count / len(self.latencies)) * 100
            distribution.append((label, count, percentage))

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
            if self.config.redis.ssl:
                pool_kwargs["ssl"] = True

            self._connection_pool = redis.ConnectionPool(**pool_kwargs)

        return self._connection_pool

    def _get_redis_client(self) -> redis.Redis:
        """Get Redis client from connection pool."""
        pool = self._create_connection_pool()
        return redis.Redis(connection_pool=pool)

    def _execute_single_query(self, executor: BaseQueryExecutor) -> Dict[str, Any]:
        """Execute a single query with error handling."""
        redis_client = self._get_redis_client()
        try:
            return executor.execute_query(redis_client)
        except Exception as e:
            return {
                "error": str(e),
                "latency_ms": 0,
                "result": None,
                "metadata": {}
            }
        finally:
            redis_client.close()

    def _run_warmup(self, executor: BaseQueryExecutor) -> None:
        """Run warmup queries to prepare connections and caches."""
        if self.config.warmup_requests <= 0:
            return

        self.console.print(f"Running {self.config.warmup_requests} warmup queries...")

        with ThreadPoolExecutor(max_workers=min(self.config.workers, 8)) as executor_pool:
            futures = [
                executor_pool.submit(self._execute_single_query, executor)
                for _ in range(self.config.warmup_requests)
            ]

            for future in as_completed(futures):
                try:
                    future.result()  # Just consume the result
                except Exception:
                    pass  # Ignore warmup errors

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

        results = []
        latencies = []
        errors = []

        start_time = time.time()

        # Variables for efficient running average calculation
        total_latency = 0.0
        successful_count = 0

        # Custom progress bar with real-time metrics
        with Progress(
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
            TimeRemainingColumn(),
            console=self.console,
            expand=True
        ) as progress:
            task = progress.add_task(
                "Benchmarking...",
                total=self.config.total_requests,
                avg_latency=0.0,
                current_qps=0.0
            )

            with ThreadPoolExecutor(max_workers=self.config.workers) as thread_executor:
                futures_queue = deque()
                submission_complete = threading.Event()

                def submit_futures():
                    """Submit futures with QPS limiting in a separate thread."""
                    qps = self.config.qps
                    next_submit_time = time.time()
                    for i in range(self.config.total_requests):
                        now = time.time()
                        if qps:
                            # Wait until the next allowed submission time
                            if now < next_submit_time:
                                time.sleep(next_submit_time - now)
                            next_submit_time = max(next_submit_time + 1.0 / qps, time.time())

                        future = thread_executor.submit(self._execute_single_query, executor)
                        futures_queue.append(future)

                    submission_complete.set()

                # Start submission thread
                submission_thread = threading.Thread(target=submit_futures)
                submission_thread.start()

                # Process completions as they become available
                completed_count = 0
                processed_futures = set()

                while completed_count < self.config.total_requests:
                    # Collect available futures
                    available_futures = []
                    while futures_queue:
                        try:
                            future = futures_queue.popleft()
                            if future not in processed_futures:
                                available_futures.append(future)
                        except IndexError:
                            break

                    # Process any completed futures
                    if available_futures:
                        # Check which futures are done (non-blocking)
                        for future in available_futures:
                            if future.done() and future not in processed_futures:
                                processed_futures.add(future)
                                try:
                                    query_result = future.result(timeout=self.config.timeout)

                                    if "error" in query_result:
                                        errors.append(query_result["error"])
                                    else:
                                        results.append(query_result["result"])
                                        latencies.append(query_result["latency_ms"])
                                        # Update running average efficiently
                                        total_latency += query_result["latency_ms"]
                                        successful_count += 1

                                    # Calculate real-time metrics
                                    elapsed_time = time.time() - start_time

                                    # Calculate average latency using running totals
                                    if successful_count > 0:
                                        avg_latency = total_latency / successful_count
                                    else:
                                        avg_latency = 0.0

                                    if elapsed_time > 0:
                                        current_qps = successful_count / elapsed_time
                                    else:
                                        current_qps = 0.0

                                    # Update progress bar with new metrics
                                    progress.advance(task)
                                    progress.update(task, avg_latency=avg_latency, current_qps=current_qps)
                                    completed_count += 1

                                except Exception as e:
                                    errors.append(str(e))
                                    elapsed_time = time.time() - start_time

                                    # Calculate metrics even on error
                                    if successful_count > 0:
                                        avg_latency = total_latency / successful_count
                                    else:
                                        avg_latency = 0.0

                                    if elapsed_time > 0:
                                        current_qps = successful_count / elapsed_time
                                    else:
                                        current_qps = 0.0

                                    progress.advance(task)
                                    progress.update(task, avg_latency=avg_latency, current_qps=current_qps)
                                    completed_count += 1
                            else:
                                # Put back unfinished futures for next iteration
                                futures_queue.appendleft(future)

                    # Small sleep to avoid busy waiting, but only if no futures completed this iteration
                    if not any(f.done() for f in available_futures):
                        time.sleep(0.01)

                # Wait for submission thread to complete
                submission_thread.join()

        end_time = time.time()
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
            latencies=latencies,
            errors=errors,
            metadata={
                "query_type": self.config.query_type,
                "workers": self.config.workers,
                "redis_host": f"{self.config.redis.host}:{self.config.redis.port}",
            },
            config=self.config
        )

    def format_results(self, results: BenchmarkResults) -> str:
        """Format results for console output."""
        output = []
        output.append("Redis Query Benchmarker Results")
        output.append("=" * 40)
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
                },
                "latency_stats": results.get_latency_stats(),
                "latency_distribution": [
                    {"bucket": label, "count": count, "percentage": percentage}
                    for label, count, percentage in results.get_latency_distribution()
                ],
                "raw_latencies": results.latencies,
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