"""Main Redis benchmarker implementation."""

import time
import json
import csv
import sys
import threading
import queue
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Optional, Tuple
import random
from dataclasses import dataclass
import redis
import numpy as np
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, TaskID, BarColumn, TextColumn, TimeRemainingColumn, SpinnerColumn

from .config import BenchmarkConfig
from .executors import BaseQueryExecutor, get_query_executor


class OnlineStatsCalculator:
    """Calculates statistics incrementally without storing all values."""

    def __init__(self):
        self.count = 0
        self.min_val = float('inf')
        self.max_val = float('-inf')
        self.sum_val = 0.0
        self.sum_sq = 0.0
        self.values_for_percentiles = []  # Only keep sample for percentiles
        self.max_samples = 10000  # Limit sample size for percentiles

        # Histogram buckets for distribution
        self.bins = [0, 100, 200, 500, 1000, 2000, float("inf")]
        self.bin_counts = [0] * (len(self.bins) - 1)

    def add_value(self, value: float):
        """Add a value and update statistics."""
        self.count += 1
        self.min_val = min(self.min_val, value)
        self.max_val = max(self.max_val, value)
        self.sum_val += value
        self.sum_sq += value * value

        # Update histogram
        for i in range(len(self.bins) - 1):
            if self.bins[i] <= value < self.bins[i + 1]:
                self.bin_counts[i] += 1
                break

        # Keep sample for percentiles (reservoir sampling)
        if len(self.values_for_percentiles) < self.max_samples:
            self.values_for_percentiles.append(value)
        else:
            # Replace random element
            idx = random.randint(0, self.count - 1)
            if idx < self.max_samples:
                self.values_for_percentiles[idx] = value

    def get_stats(self) -> Dict[str, float]:
        """Get computed statistics."""
        if self.count == 0:
            return {}

        mean = self.sum_val / self.count
        variance = (self.sum_sq / self.count) - (mean * mean)
        std_dev = variance ** 0.5 if variance > 0 else 0.0

        stats = {
            "count": self.count,
            "average": mean,
            "min": self.min_val if self.min_val != float('inf') else 0.0,
            "max": self.max_val if self.max_val != float('-inf') else 0.0,
            "std_dev": std_dev,
        }

        # Calculate percentiles from sample
        if self.values_for_percentiles:
            sample_array = np.array(self.values_for_percentiles)
            stats.update({
                "median": float(np.median(sample_array)),
                "p90": float(np.percentile(sample_array, 90)),
                "p95": float(np.percentile(sample_array, 95)),
                "p99": float(np.percentile(sample_array, 99)),
            })

        return stats

    def get_distribution(self) -> List[Tuple[str, int, float]]:
        """Get latency distribution."""
        if self.count == 0:
            return []

        labels = ["<100ms", "100-200ms", "200-500ms", "500ms-1s", "1-2s", ">2s"]
        distribution = []
        for count, label in zip(self.bin_counts, labels):
            percentage = (count / self.count) * 100
            distribution.append((label, count, percentage))

        return distribution


class ThreadSafeCounters:
    """Thread-safe counters for tracking benchmark progress."""

    def __init__(self):
        self._lock = threading.Lock()
        self.completed_count = 0
        self.successful_count = 0
        self.total_latency = 0.0
        self.total_result_count = 0
        self.errors_count = 0

    def update_success(self, latency_ms: float, result_count: int) -> None:
        """Update counters for a successful query."""
        with self._lock:
            self.completed_count += 1
            self.successful_count += 1
            self.total_latency += latency_ms
            self.total_result_count += result_count

    def update_error(self) -> None:
        """Update counters for a failed query."""
        with self._lock:
            self.completed_count += 1
            self.errors_count += 1

    def get_snapshot(self) -> Dict[str, Any]:
        """Get a thread-safe snapshot of current counters."""
        with self._lock:
            return {
                'completed_count': self.completed_count,
                'successful_count': self.successful_count,
                'total_latency': self.total_latency,
                'total_result_count': self.total_result_count,
                'errors_count': self.errors_count
            }


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

    # Statistics computed during benchmark to avoid storing raw data
    latency_stats: Optional[Dict[str, float]] = None
    latency_distribution: Optional[List[Tuple[str, int, float]]] = None

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
        # Return precomputed stats if available to avoid recalculation
        if self.latency_stats is not None:
            return self.latency_stats

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
        # Return precomputed distribution if available
        if self.latency_distribution is not None:
            return self.latency_distribution

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
        self.is_interactive = sys.stdout.isatty()
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
                "protocol": self.config.redis.protocol,
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

    def _prewarm_connection_pool_direct(self, min_connections: int) -> None:
        """
        Pre-warm the connection pool using direct pool connection management.

        This method creates the specified number of connections by directly
        using the pool's get_connection() method, exercises each connection
        with a PING command, and then releases them back to the pool.

        Args:
            min_connections: Number of connections to pre-create and warm
        """
        if min_connections <= 0:
            return

        connection_pool = self._create_connection_pool()
        raw_connections = []

        try:
            self.console.print(f"Pre-warming {min_connections} connections...")

            # Directly get connections from pool
            for i in range(min_connections):
                try:
                    # get_connection() requires a command_name parameter
                    conn = connection_pool.get_connection("PING")
                    conn.connect()  # Establish the connection

                    # Exercise the connection with PING
                    conn.send_command("PING")
                    response = conn.read_response()

                    if response != b"PONG" and response != "PONG":
                        self.console.print(f"Warning: Unexpected PING response: {response}")

                    # Optional: test index access if configured
                    if self.config.index_name:
                        try:
                            conn.send_command("FT.INFO", self.config.index_name)
                            conn.read_response()  # Read and discard the response
                        except Exception:
                            # Index might not exist or command might fail, continue anyway
                            pass

                    raw_connections.append(conn)

                    if self.config.verbose:
                        self.console.print(f"✓ Pre-warmed connection {i+1}/{min_connections}")

                except Exception as e:
                    self.console.print(f"Warning: Failed to pre-warm connection {i+1}: {e}")
                    break

            created_count = len(raw_connections)
            if created_count > 0:
                self.console.print(f"Successfully pre-warmed {created_count} connections")
            else:
                self.console.print("Warning: No connections were pre-warmed")

        finally:
            # Release all connections back to pool
            for conn in raw_connections:
                try:
                    connection_pool.release(conn)
                except Exception as e:
                    if self.config.verbose:
                        self.console.print(f"Warning: Failed to release connection: {e}")

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

    def _progress_updater_thread(self, progress: Progress, task: TaskID, counters: ThreadSafeCounters,
                                 start_time: float, stop_event: threading.Event, qps_controller=None) -> None:
        """Background thread that automatically updates progress bar with adaptive frequency."""
        # More responsive update frequency for better progress tracking
        if self.config.total_requests <= 100:
            update_interval = 0.1  # Very frequent updates for small test runs
        elif self.config.total_requests <= 1000:
            update_interval = 0.2  # Frequent updates for small runs
        else:
            update_interval = 0.5 if self.config.total_requests < 100000 else 1.0

        last_completed_count = 0

        while not stop_event.is_set():
            try:
                # Get current snapshot of counters
                snapshot = counters.get_snapshot()

                # Calculate metrics from snapshot
                metrics = self._calculate_metrics(
                    start_time,
                    snapshot['successful_count'],
                    snapshot['total_latency'],
                    snapshot['total_result_count']
                )

                # Get current target QPS if jitter controller is available
                current_target_qps = None
                if qps_controller:
                    if hasattr(qps_controller, 'get_current_target_qps'):
                        current_target_qps = qps_controller.get_current_target_qps()
                    elif hasattr(qps_controller, 'target_qps'):
                        current_target_qps = qps_controller.target_qps

                # Always update if there's been progress or if this is the first update
                should_update = (snapshot['completed_count'] != last_completed_count or 
                               last_completed_count == 0)

                # Debug: Always show first few updates to understand timing
                elapsed = time.time() - start_time
                if elapsed < 10.0 or should_update:  # Show all updates in first 10 seconds, then only changes
                    if self.is_interactive:
                        # Update progress bar for interactive terminals
                        update_fields = {
                            "avg_latency": metrics["avg_latency"],
                            "current_qps": metrics["current_qps"]
                        }

                        # Add target QPS if available
                        if current_target_qps is not None:
                            update_fields["target_qps"] = current_target_qps

                        if self.config.show_expanded_metrics:
                            update_fields.update({
                                "avg_results": metrics["avg_results"],
                                "norm_latency": metrics["norm_latency"]
                            })

                        progress.update(task, completed=snapshot['completed_count'], **update_fields)
                    else:
                        # Print status messages for non-interactive terminals
                        percentage = (snapshot['completed_count'] / self.config.total_requests) * 100
                        status_msg = f"Progress: {snapshot['completed_count']}/{self.config.total_requests} ({percentage:.1f}%) | Avg Latency: {metrics['avg_latency']:.1f}ms | QPS: {metrics['current_qps']:.1f}"
                        
                        # Add target QPS if available (shows jitter in action)
                        if current_target_qps is not None:
                            status_msg += f" | Target: {current_target_qps:.1f}"
                        
                        status_msg += f" | t={elapsed:.2f}s"

                        if self.config.show_expanded_metrics:
                            status_msg += f" | Avg Results: {metrics['avg_results']:.1f} | Norm Latency: {metrics['norm_latency']:.2f}ms/result"

                        print(status_msg, flush=True)

                    if should_update:
                        last_completed_count = snapshot['completed_count']

                # Adaptive update frequency
                stop_event.wait(update_interval)

            except Exception:
                # Ignore any errors in progress updates to avoid breaking the benchmark
                stop_event.wait(0.5)

    def _run_qps_controlled_benchmark(
        self,
        thread_executor: ThreadPoolExecutor,
        executor: BaseQueryExecutor,
        connection_pool: redis.ConnectionPool,
        total_requests: int,
        qps_controller: 'QpsController',
        latency_stats_calc: OnlineStatsCalculator,
        latencies_sample: deque,
        result_counts_sample: deque,
        errors: deque,
        counters: ThreadSafeCounters
    ) -> None:
        """
        Run QPS-controlled benchmark with intelligent batch submission.

        The QPS limiting is enforced by submitting queries in calculated batches at timed intervals,
        ensuring the target QPS is respected while maintaining worker utilization and connection
        pool efficiency.
        """

        # Calculate optimal batch submission strategy
        initial_target_qps = qps_controller.get_current_target_qps() if hasattr(qps_controller, 'get_current_target_qps') else qps_controller.target_qps
        max_workers = self.config.workers

        # Determine initial batch size and interval based on QPS and worker count
        if initial_target_qps >= max_workers:
            # High QPS: Submit batches frequently to maintain rate
            base_batch_size = max(1, min(max_workers, int(initial_target_qps / 10)))  # 10 batches per second
        else:
            # Low QPS: Submit smaller batches less frequently
            base_batch_size = 1

        # Process queries with controlled submission and concurrent result processing
        all_futures = []
        remaining_requests = total_requests
        next_batch_time = time.time()
        submitted_count = 0

        # Use a separate thread to handle result processing concurrently
        import threading
        from queue import Queue
        
        result_queue = Queue()
        processing_complete = threading.Event()
        
        def result_processor():
            """Process results as they complete, updating counters immediately."""
            processed = 0
            while processed < total_requests:
                try:
                    future = result_queue.get(timeout=1.0)
                    try:
                        query_result = future.result(timeout=self.config.timeout)
                        qps_controller.record_completion()

                        if "error" in query_result:
                            errors.append(query_result["error"])
                            counters.update_error()
                        else:
                            # Update incremental statistics
                            latency_ms = query_result["latency_ms"]
                            latency_stats_calc.add_value(latency_ms)

                            # Keep samples for export
                            latencies_sample.append(latency_ms)

                            # Extract result count from metadata
                            result_count = self._extract_result_count(query_result)
                            result_counts_sample.append(result_count)

                            # Update thread-safe counters
                            counters.update_success(latency_ms, result_count)

                    except Exception as e:
                        errors.append(str(e))
                        counters.update_error()
                    
                    processed += 1
                    result_queue.task_done()
                except:
                    # Timeout waiting for results - continue checking
                    continue
            
            processing_complete.set()

        # Start result processor thread
        processor_thread = threading.Thread(target=result_processor, daemon=True)
        processor_thread.start()

        # Submit queries in controlled batches while results are processed concurrently
        while remaining_requests > 0:
            # Get current target QPS (may have changed due to jitter)
            current_target_qps = qps_controller.get_current_target_qps() if hasattr(qps_controller, 'get_current_target_qps') else qps_controller.target_qps
            
            # Recalculate batch size and interval based on current target QPS
            if current_target_qps >= max_workers:
                batch_size = max(1, min(max_workers, int(current_target_qps / 10)))
                batch_interval = batch_size / current_target_qps
            else:
                batch_size = 1
                batch_interval = 1.0 / current_target_qps
            
            # Calculate current batch size (might be smaller for last batch)
            current_batch_size = min(batch_size, remaining_requests)
            
            # Check adaptive rate limiting before submission
            if not qps_controller.should_submit_batch(current_batch_size):
                # Apply adaptive backpressure - skip this submission cycle
                adaptive_delay = qps_controller.calculate_adaptive_delay(batch_interval)
                time.sleep(adaptive_delay)
                continue
                
            # Wait until it's time for the next batch
            current_time = time.time()
            if current_time < next_batch_time:
                base_sleep_time = next_batch_time - current_time
                # Apply adaptive delay adjustment
                adaptive_sleep_time = qps_controller.calculate_adaptive_delay(base_sleep_time)
                time.sleep(adaptive_sleep_time)

            # Submit the batch of queries and add to result queue immediately
            for _ in range(current_batch_size):
                future = thread_executor.submit(self._execute_single_query, executor, connection_pool)
                all_futures.append(future)
                result_queue.put(future)  # Add to queue for immediate processing

            remaining_requests -= current_batch_size
            submitted_count += current_batch_size

            # Calculate next batch time (base interval, adaptive adjustment applied above)
            next_batch_time = time.time() + batch_interval

        # Wait for all results to be processed
        processing_complete.wait()

    class AdaptiveQpsWrapper:
        """
        Adaptive wrapper that enhances any QPS controller with completion-aware rate limiting.
        Prevents QPS overshooting after latency spikes by tracking completion patterns and
        applying backpressure when completion rates exceed targets.
        """

        def __init__(self, base_controller, tolerance_percent: float = 10.0, window_size: int = 50):
            self.base_controller = base_controller
            self.tolerance_percent = tolerance_percent
            self.tolerance_factor = 1.0 + (tolerance_percent / 100.0)
            
            # Get target QPS to optimize parameters for low QPS scenarios
            target_qps = self.get_base_target_qps()
            
            # Auto-optimize parameters for low QPS (≤5 QPS)
            if target_qps <= 5.0:
                # Smaller window for faster reaction at low QPS
                window_size = min(window_size, max(10, int(target_qps * 10)))  # 10-50 samples
                # More sensitive tolerance for low QPS
                if tolerance_percent == 10.0:  # Only adjust if using default
                    self.tolerance_percent = max(5.0, tolerance_percent / 2)  # 5% tolerance
                    self.tolerance_factor = 1.0 + (self.tolerance_percent / 100.0)
                # Reduced minimum completion threshold
                self.min_completions_threshold = max(2, int(target_qps))  # 2-5 completions
            else:
                self.min_completions_threshold = 5  # Standard threshold for higher QPS
            
            # Completion rate tracking
            self.completion_times = deque(maxlen=window_size)
            self.submission_debt = 0.0  # Track submission vs completion imbalance
            self.lock = threading.Lock()
            
            # Adaptive parameters optimized for target QPS
            if target_qps <= 1.0:
                # Very low QPS: smaller delays, higher sensitivity
                self.min_submission_delay = 0.01   # 10ms minimum
                self.max_submission_delay = 2.0    # 2s maximum
                self.debt_reduction_rate = 0.8     # Faster debt reduction
            elif target_qps <= 5.0:
                # Low QPS: moderate delays
                self.min_submission_delay = 0.005  # 5ms minimum
                self.max_submission_delay = 3.0    # 3s maximum
                self.debt_reduction_rate = 0.6     # Moderate debt reduction
            else:
                # Standard QPS: original parameters
                self.min_submission_delay = 0.001  # 1ms minimum
                self.max_submission_delay = 5.0    # 5s maximum
                self.debt_reduction_rate = 0.5     # Standard debt reduction
            
        def get_current_target_qps(self) -> float:
            """Delegate to base controller."""
            if hasattr(self.base_controller, 'get_current_target_qps'):
                return self.base_controller.get_current_target_qps()
            elif hasattr(self.base_controller, 'target_qps'):
                return self.base_controller.target_qps
            return 0.0
            
        def record_completion(self):
            """Record completion and update adaptive state."""
            with self.lock:
                current_time = time.time()
                self.completion_times.append(current_time)
                
                # Reduce submission debt when completions occur (rate varies by QPS)
                self.submission_debt = max(0.0, self.submission_debt - self.debt_reduction_rate)
                
            # Delegate to base controller
            self.base_controller.record_completion()
            
        def should_submit_batch(self, batch_size: int) -> bool:
            """
            Determine if a batch should be submitted now based on completion patterns.
            Returns True if submission should proceed, False if backpressure should be applied.
            """
            with self.lock:
                target_qps = self.get_current_target_qps()
                if target_qps <= 0 or len(self.completion_times) < self.min_completions_threshold:
                    return True  # Not enough data, allow submission
                    
                # Calculate recent completion rate
                recent_completion_rate = self._calculate_recent_completion_rate()
                
                # Check if completion rate exceeds tolerance
                if recent_completion_rate > (target_qps * self.tolerance_factor):
                    # Apply backpressure - accumulate submission debt
                    overshoot_ratio = recent_completion_rate / target_qps
                    self.submission_debt += (overshoot_ratio - 1.0) * batch_size
                    
                    # Probabilistic backpressure based on overshoot severity
                    backpressure_probability = min(0.9, (overshoot_ratio - 1.0) / 2.0)
                    return random.random() > backpressure_probability
                    
                return True
                
        def calculate_adaptive_delay(self, base_delay: float) -> float:
            """
            Calculate adaptive submission delay based on completion patterns and submission debt.
            """
            with self.lock:
                if self.submission_debt <= 0:
                    return base_delay
                    
                # Apply exponential backoff based on submission debt
                debt_factor = min(3.0, 1.0 + (self.submission_debt / 20.0))
                adaptive_delay = base_delay * debt_factor
                
                # Clamp to reasonable bounds
                return max(self.min_submission_delay, min(self.max_submission_delay, adaptive_delay))
                
        def _calculate_recent_completion_rate(self) -> float:
            """
            Calculate recent completion rate from completion times window.
            Must be called with lock held.
            """
            if len(self.completion_times) < 2:
                return 0.0
                
            time_span = self.completion_times[-1] - self.completion_times[0]
            if time_span <= 0:
                return 0.0
                
            return (len(self.completion_times) - 1) / time_span
            
        def get_stats(self) -> Dict[str, float]:
            """Get combined stats from base controller and adaptive wrapper."""
            base_stats = self.base_controller.get_stats()
            
            with self.lock:
                recent_completion_rate = self._calculate_recent_completion_rate()
                adaptive_stats = {
                    "recent_completion_rate": recent_completion_rate,
                    "submission_debt": self.submission_debt,
                    "tolerance_factor": self.tolerance_factor,
                }
                
            # Merge stats
            base_stats.update(adaptive_stats)
            return base_stats
            
        def get_base_target_qps(self) -> float:
            """Get the base target QPS from the controller (used for initialization)."""
            if hasattr(self.base_controller, 'get_current_target_qps'):
                return self.base_controller.get_current_target_qps()
            elif hasattr(self.base_controller, 'base_qps'):
                return self.base_controller.base_qps
            elif hasattr(self.base_controller, 'target_qps'):
                return self.base_controller.target_qps
            return 10.0  # Default assumption for parameter optimization
            
        def __getattr__(self, name):
            """Delegate unknown attributes to base controller."""
            return getattr(self.base_controller, name)

    class JitteredQpsController:
        """
        QPS controller with jitter support for realistic traffic simulation.
        Periodically recalculates target QPS based on configured jitter parameters.
        """

        def __init__(self, base_qps: float, jitter_percent: float, jitter_interval: float, distribution: str = "uniform", direction: str = "random"):
            self.base_qps = base_qps
            self.jitter_percent = jitter_percent
            self.jitter_interval = jitter_interval
            self.distribution = distribution
            self.direction = direction
            
            self.current_target_qps = base_qps
            self.last_jitter_update = time.time()
            self.start_time = time.time()
            self.completed_count = 0
            self.lock = threading.Lock()
            
            # Keep track of completion times for statistics
            self.completed_times = deque(maxlen=50)
            
            # Initialize first jitter value
            self._recalculate_jitter()

        def _recalculate_jitter(self):
            """Recalculate target QPS based on jitter distribution and direction."""
            jitter_factor = self.jitter_percent / 100.0
            
            # First, calculate the base jitter multiplier based on distribution
            if self.distribution == "uniform":
                # Uniform distribution: ±jitter_percent with equal probability
                raw_jitter = random.uniform(-jitter_factor, jitter_factor)
            
            elif self.distribution == "normal":
                # Normal distribution: centered on base_qps with std_dev = jitter_percent/3
                std_dev = jitter_factor / 3.0  # 99.7% of values within ±jitter_percent
                raw_jitter = random.gauss(0, std_dev)
                # Clamp to reasonable bounds
                raw_jitter = max(-jitter_factor, min(jitter_factor, raw_jitter))
            
            elif self.distribution == "triangular":
                # Triangular distribution: more likely to be near base_qps
                raw_jitter = random.triangular(-jitter_factor, jitter_factor, 0)
            
            elif self.distribution == "bursty":
                # Bursty distribution: 80% normal traffic, 20% spikes
                if random.random() < 0.8:
                    # Normal traffic: slight variation
                    raw_jitter = random.uniform(-jitter_factor * 0.3, jitter_factor * 0.3)
                else:
                    # Traffic spike: significant increase
                    raw_jitter = random.uniform(jitter_factor * 0.5, jitter_factor)
            
            else:
                # Fallback to uniform
                raw_jitter = random.uniform(-jitter_factor, jitter_factor)
            
            # Apply direction coordination for distributed environments
            if self.direction == "positive":
                # Force all jitter to be positive (traffic spikes)
                jitter_multiplier = 1.0 + abs(raw_jitter)
            
            elif self.direction == "negative":
                # Force all jitter to be negative (traffic dips)
                jitter_multiplier = 1.0 - abs(raw_jitter)
            
            elif self.direction == "alternating":
                # Coordinated alternating pattern based on global time
                # All pods will synchronize to the same high/low phase
                current_time = time.time()
                phase_number = int(current_time // self.jitter_interval)
                is_high_phase = (phase_number % 2) == 0
                
                if is_high_phase:
                    # High phase: use positive jitter
                    jitter_multiplier = 1.0 + abs(raw_jitter)
                else:
                    # Low phase: use negative jitter
                    jitter_multiplier = 1.0 - abs(raw_jitter)
            
            else:  # "random" or default
                # Original random behavior
                jitter_multiplier = 1.0 + raw_jitter
            
            # Ensure we don't go below zero or unreasonably low QPS
            jitter_multiplier = max(0.05, jitter_multiplier)  # Allow lower minimum for distributed scenarios
            
            self.current_target_qps = self.base_qps * jitter_multiplier
            self.last_jitter_update = time.time()

        def get_current_target_qps(self) -> float:
            """Get the current target QPS, recalculating jitter if needed."""
            current_time = time.time()
            if current_time - self.last_jitter_update >= self.jitter_interval:
                self._recalculate_jitter()
            return self.current_target_qps

        def record_completion(self):
            """Record that a query was completed."""
            with self.lock:
                self.completed_count += 1
                self.completed_times.append(time.time())

        def _calculate_actual_qps(self) -> float:
            """Calculate actual QPS based on recent completions."""
            if len(self.completed_times) < 2:
                return 0.0

            time_span = self.completed_times[-1] - self.completed_times[0]
            if time_span <= 0:
                return 0.0

            return (len(self.completed_times) - 1) / time_span

        def get_stats(self) -> Dict[str, float]:
            """Get current controller statistics."""
            with self.lock:
                elapsed = time.time() - self.start_time
                completed_qps = self.completed_count / elapsed if elapsed > 0 else 0
                actual_qps = self._calculate_actual_qps()

                return {
                    "base_qps": self.base_qps,
                    "current_target_qps": self.current_target_qps,
                    "completed_qps": completed_qps,
                    "actual_qps": actual_qps,
                    "completed_count": self.completed_count,
                    "jitter_percent": self.jitter_percent,
                    "jitter_distribution": self.distribution,
                    "jitter_direction": self.direction
                }

    class QpsController:
        """
        QPS controller for tracking completion statistics.
        Rate limiting is handled by controlling query submission timing.
        """

        def __init__(self, target_qps: float):
            self.target_qps = target_qps
            self.start_time = time.time()
            self.completed_count = 0
            self.lock = threading.Lock()

            # Keep track of completion times for statistics
            self.completed_times = deque(maxlen=50)

        def record_completion(self):
            """Record that a query was completed."""
            with self.lock:
                self.completed_count += 1
                self.completed_times.append(time.time())

        def _calculate_actual_qps(self) -> float:
            """Calculate actual QPS based on recent completions."""
            if len(self.completed_times) < 2:
                return 0.0

            time_span = self.completed_times[-1] - self.completed_times[0]
            if time_span <= 0:
                return 0.0

            return (len(self.completed_times) - 1) / time_span

        def get_stats(self) -> Dict[str, float]:
            """Get current controller statistics."""
            with self.lock:
                elapsed = time.time() - self.start_time
                completed_qps = self.completed_count / elapsed if elapsed > 0 else 0
                actual_qps = self._calculate_actual_qps()

                return {
                    "target_qps": self.target_qps,
                    "completed_qps": completed_qps,
                    "actual_qps": actual_qps,
                    "completed_count": self.completed_count
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

        # Pre-warm connection pool if configured
        if self.config.pre_warm_connections > 0:
            self._prewarm_connection_pool_direct(self.config.pre_warm_connections)

        # Run warmup
        self._run_warmup(executor)

        # Run main benchmark
        self.console.print(f"Starting benchmark with {self.config.total_requests} requests using {self.config.workers} workers...")

        # Use incremental statistics to avoid storing all data in memory
        latency_stats_calc = OnlineStatsCalculator()

        # Dramatically reduce memory usage for large benchmarks
        # Use adaptive sampling - fewer samples for very large tests
        if self.config.total_requests >= 100000:
            max_raw_samples = min(1000, self.config.total_requests // 1000)  # Much smaller sample
        else:
            max_raw_samples = min(5000, self.config.total_requests // 10)

        latencies_sample = deque(maxlen=max_raw_samples)
        result_counts_sample = deque(maxlen=max_raw_samples)

        # Keep limited error samples
        errors = deque(maxlen=1000)  # Limit error storage

        start_time = time.time()

        # Thread-safe counters for progress tracking
        counters = ThreadSafeCounters()

        # Initialize QPS controller if QPS limiting is enabled (needed for progress bar setup)
        qps_controller = None
        if self.config.qps:
            # Create base controller
            if self.config.jitter_enabled:
                base_controller = self.JitteredQpsController(
                    self.config.qps,
                    self.config.jitter_percent,
                    self.config.jitter_interval_secs,
                    self.config.jitter_distribution,
                    self.config.jitter_direction
                )
                controller_description = f"jittered QPS controller: {self.config.qps}±{self.config.jitter_percent}% ({self.config.jitter_distribution} distribution, {self.config.jitter_direction} direction)"
            else:
                base_controller = self.QpsController(self.config.qps)
                controller_description = f"QPS controller: {self.config.qps}"
                
            # Wrap with adaptive completion rate limiting
            qps_controller = self.AdaptiveQpsWrapper(
                base_controller, 
                tolerance_percent=self.config.qps_tolerance_percent
            )
            
            if self.config.verbose:
                tolerance_msg = f"(±{self.config.qps_tolerance_percent:.1f}% tolerance)"
                self.console.print(f"[cyan]Using adaptive {controller_description} with completion rate limiting {tolerance_msg}[/cyan]")

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

        # Add target QPS column if QPS limiting is enabled
        if qps_controller:
            progress_columns.extend([
                TextColumn("•"),
                TextColumn("[bold yellow]Target: {task.fields[target_qps]:.1f}"),
            ])

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

        # For non-interactive terminals, disable progress bar display
        progress_kwargs = {
            "console": self.console,
            "expand": True
        }
        if not self.is_interactive:
            progress_kwargs["disable"] = True
            print("Starting benchmark progress tracking...", flush=True)

        with Progress(
                *progress_columns,
                **progress_kwargs
            ) as progress:
                task_fields = {
                    "avg_latency": 0.0,
                    "current_qps": 0.0
                }

                # Add target QPS field if QPS limiting is enabled
                if qps_controller:
                    initial_target = qps_controller.get_current_target_qps() if hasattr(qps_controller, 'get_current_target_qps') else qps_controller.target_qps
                    task_fields["target_qps"] = initial_target

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

                # Start background progress updater thread
                stop_progress_event = threading.Event()
                progress_thread = threading.Thread(
                    target=self._progress_updater_thread,
                    args=(progress, task, counters, start_time, stop_progress_event, qps_controller),
                    daemon=True
                )
                progress_thread.start()

                # Use ThreadPoolExecutor with batch future management for large query counts
                with ThreadPoolExecutor(max_workers=self.config.workers) as thread_executor:
                    try:
                        # Optimize for large query counts by batching future submission
                        batch_size = min(10000, self.config.total_requests)
                        remaining_requests = self.config.total_requests
                        all_futures = []

                        # QPS controller already initialized above

                        # Process queries efficiently with proper threading
                        if qps_controller:
                            # QPS-controlled submission with efficient async processing
                            self._run_qps_controlled_benchmark(
                                thread_executor, executor, connection_pool, remaining_requests,
                                qps_controller, latency_stats_calc, latencies_sample,
                                result_counts_sample, errors, counters
                            )
                        else:
                            # Original high-performance batch processing for unlimited QPS
                            while remaining_requests > 0:
                                current_batch_size = min(batch_size, remaining_requests)

                                # Submit batch of futures
                                batch_futures = [
                                    thread_executor.submit(self._execute_single_query, executor, connection_pool)
                                    for _ in range(current_batch_size)
                                ]

                                # Process batch results asynchronously
                                for future in as_completed(batch_futures):
                                    try:
                                        query_result = future.result(timeout=self.config.timeout)

                                        if "error" in query_result:
                                            errors.append(query_result["error"])
                                            counters.update_error()
                                        else:
                                            # Update incremental statistics
                                            latency_ms = query_result["latency_ms"]
                                            latency_stats_calc.add_value(latency_ms)

                                            # Keep samples for export
                                            latencies_sample.append(latency_ms)

                                            # Extract result count from metadata
                                            result_count = self._extract_result_count(query_result)
                                            result_counts_sample.append(result_count)

                                            # Update thread-safe counters (progress thread will read these)
                                            counters.update_success(latency_ms, result_count)

                                    except Exception as e:
                                        errors.append(str(e))
                                        counters.update_error()

                                remaining_requests -= current_batch_size

                        # Final progress update to ensure completion is shown
                        final_snapshot = counters.get_snapshot()
                        final_metrics = self._calculate_metrics(
                            start_time,
                            final_snapshot['successful_count'],
                            final_snapshot['total_latency'],
                            final_snapshot['total_result_count']
                        )

                        # Get final target QPS
                        final_target_qps = None
                        if qps_controller:
                            if hasattr(qps_controller, 'get_current_target_qps'):
                                final_target_qps = qps_controller.get_current_target_qps()
                            elif hasattr(qps_controller, 'target_qps'):
                                final_target_qps = qps_controller.target_qps

                        if self.is_interactive:
                            final_update_fields = {
                                "avg_latency": final_metrics["avg_latency"],
                                "current_qps": final_metrics["current_qps"]
                            }
                            
                            if final_target_qps is not None:
                                final_update_fields["target_qps"] = final_target_qps
                                
                            if self.config.show_expanded_metrics:
                                final_update_fields.update({
                                    "avg_results": final_metrics["avg_results"],
                                    "norm_latency": final_metrics["norm_latency"]
                                })
                            progress.update(task, completed=final_snapshot['completed_count'], **final_update_fields)
                        else:
                            # Final status message for non-interactive terminals
                            final_percentage = (final_snapshot['completed_count'] / self.config.total_requests) * 100
                            final_status_msg = f"Progress: {final_snapshot['completed_count']}/{self.config.total_requests} ({final_percentage:.1f}%) | Avg Latency: {final_metrics['avg_latency']:.1f}ms | QPS: {final_metrics['current_qps']:.1f}"
                            
                            if final_target_qps is not None:
                                final_status_msg += f" | Target: {final_target_qps:.1f}"
                                
                            if self.config.show_expanded_metrics:
                                final_status_msg += f" | Avg Results: {final_metrics['avg_results']:.1f} | Norm Latency: {final_metrics['norm_latency']:.2f}ms/result"
                            print(final_status_msg, flush=True)

                        # Stop progress updater thread when benchmark completes normally
                        stop_progress_event.set()

                    except KeyboardInterrupt:
                        interrupted = True
                        interrupt_time = time.time()

                        # Print message immediately
                        self.console.print("\n[yellow]Benchmark interrupted by user. Collecting partial results...[/yellow]")

                        # Note: Individual futures will be cancelled by ThreadPoolExecutor.__exit__

                        # Stop progress updater thread
                        stop_progress_event.set()

                        # Stop progress display
                        try:
                            progress.stop()
                        except:
                            pass

        # Calculate end time - use interrupt time if interrupted, otherwise current time
        end_time = interrupt_time if interrupted else time.time()
        total_time = end_time - start_time

        # Print QPS controller stats if enabled
        if qps_controller and self.config.verbose:
            stats = qps_controller.get_stats()
            self.console.print(f"\n[cyan]Adaptive QPS Controller Stats:[/cyan]")
            
            if 'base_qps' in stats:
                # Jittered controller stats
                self.console.print(f"  Base QPS: {stats['base_qps']:.2f}")
                self.console.print(f"  Current Target QPS: {stats['current_target_qps']:.2f}")
                self.console.print(f"  Jitter: ±{stats['jitter_percent']:.1f}% ({stats['jitter_distribution']}, {stats['jitter_direction']} direction)")
            else:
                # Regular controller stats
                self.console.print(f"  Target QPS: {stats['target_qps']:.2f}")
            
            self.console.print(f"  Actual QPS: {stats['actual_qps']:.2f}")
            self.console.print(f"  Completed QPS: {stats['completed_qps']:.2f}")
            self.console.print(f"  Recent Completion Rate: {stats.get('recent_completion_rate', 0.0):.2f}")
            self.console.print(f"  Submission Debt: {stats.get('submission_debt', 0.0):.1f}")
            self.console.print(f"  Total Completed: {stats['completed_count']}")
            
            # Show adaptive effectiveness
            target = stats.get('current_target_qps', stats.get('target_qps', 0))
            recent_rate = stats.get('recent_completion_rate', 0)
            if target > 0 and recent_rate > 0:
                overshoot_pct = ((recent_rate - target) / target) * 100
                if abs(overshoot_pct) > 1:
                    self.console.print(f"  Rate Variance: {overshoot_pct:+.1f}%")

        # Cleanup resources to prevent memory leaks
        cleanup_client = self._get_redis_client()
        try:
            executor.cleanup(cleanup_client)
        finally:
            cleanup_client.close()

        # Close connection pool to release all connections
        if self._connection_pool:
            self._connection_pool.disconnect()
            self._connection_pool = None

        # Get final statistics from incremental calculator
        final_latency_stats = latency_stats_calc.get_stats()
        final_latency_distribution = latency_stats_calc.get_distribution()

        successful_requests = latency_stats_calc.count
        failed_requests = len(errors)

        return BenchmarkResults(
            total_requests=self.config.total_requests,
            successful_requests=successful_requests,
            failed_requests=failed_requests,
            total_time=total_time,
            latencies=list(latencies_sample),  # Only sample data
            errors=list(errors),
            metadata={
                "query_type": self.config.query_type,
                "workers": self.config.workers,
                "redis_host": f"{self.config.redis.host}:{self.config.redis.port}",
                "interrupted": interrupted,
                "completed_requests": successful_requests + failed_requests,
                "note": f"Latency data is sampled (max {max_raw_samples} samples) to prevent OOM",
                "qps_stats": qps_controller.get_stats() if qps_controller else None,
            },
            config=self.config,
            result_counts=list(result_counts_sample),
            latency_stats=final_latency_stats,
            latency_distribution=final_latency_distribution
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
                "raw_latencies": results.latencies if len(results.latencies) <= 10000 else [],
                "raw_result_counts": results.result_counts if len(results.result_counts) <= 10000 else [],
                "sampling_note": "Raw data excluded for large datasets to prevent memory issues" if len(results.latencies) > 10000 else None,
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