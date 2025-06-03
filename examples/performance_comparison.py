#!/usr/bin/env python3
"""
Example: Performance Comparison

This script demonstrates how to compare the performance of different
Redis search configurations and query types systematically.
"""

import json
import time
from pathlib import Path
from redis_benchmarker import BenchmarkConfig, RedisBenchmarker


def run_comparison_test(test_name: str, configs: list, output_dir: str = "results"):
    """Run a series of benchmark tests and save results."""

    print(f"\n{'='*60}")
    print(f"Running: {test_name}")
    print(f"{'='*60}")

    results = []
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    for i, config in enumerate(configs):
        print(f"\nTest {i+1}/{len(configs)}: {config.query_type}")
        print(f"Workers: {config.workers}, Requests: {config.total_requests}")
        print("-" * 40)

        try:
            benchmarker = RedisBenchmarker(config)
            result = benchmarker.run_benchmark()

            # Print summary
            print(f"QPS: {result.qps:.2f}")
            print(f"Avg Latency: {result.get_latency_stats().get('average', 0):.2f}ms")
            print(f"P95 Latency: {result.get_latency_stats().get('p95', 0):.2f}ms")
            print(f"Success Rate: {result.success_rate:.1f}%")

            # Save detailed results
            output_file = output_path / f"{test_name.lower().replace(' ', '_')}_test_{i+1}.json"
            benchmarker.save_results(result, str(output_file), "json")

            results.append({
                "test_name": f"{test_name} - Test {i+1}",
                "config": config.to_dict(),
                "qps": result.qps,
                "avg_latency_ms": result.get_latency_stats().get('average', 0),
                "p95_latency_ms": result.get_latency_stats().get('p95', 0),
                "p99_latency_ms": result.get_latency_stats().get('p99', 0),
                "success_rate": result.success_rate,
                "total_time": result.total_time
            })

        except Exception as e:
            print(f"Error: {e}")
            results.append({
                "test_name": f"{test_name} - Test {i+1}",
                "config": config.to_dict(),
                "error": str(e)
            })

    # Save summary
    summary_file = output_path / f"{test_name.lower().replace(' ', '_')}_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_dir}/")
    return results


def worker_scaling_test():
    """Test how performance scales with different worker counts."""

    worker_counts = [1, 2, 4, 8, 16, 32, 64]
    configs = []

    for workers in worker_counts:
        config = BenchmarkConfig(
            total_requests=500,
            workers=workers,
            query_type="vector_search",
            index_name="sample_index",
            vector_dim=1536,
            num_results=10
        )
        configs.append(config)

    return run_comparison_test("Worker Scaling", configs)


def query_type_comparison():
    """Compare different query types."""

    query_types = [
        ("vector_search", {}),
        ("hybrid_search", {"filter_expression": "@category:{electronics}"}),
        ("redis_py", {"search_query": "@category:{electronics}"})
    ]

    configs = []
    for query_type, extra_params in query_types:
        config = BenchmarkConfig(
            total_requests=1000,
            workers=16,
            query_type=query_type,
            index_name="sample_index",
            vector_dim=1536,
            num_results=10,
            extra_params=extra_params
        )
        configs.append(config)

    return run_comparison_test("Query Type Comparison", configs)


def vector_dimension_test():
    """Test performance with different vector dimensions."""

    dimensions = [128, 256, 512, 1024, 1536, 2048]
    configs = []

    for dim in dimensions:
        config = BenchmarkConfig(
            total_requests=500,
            workers=16,
            query_type="vector_search",
            index_name="sample_index",
            vector_dim=dim,
            num_results=10
        )
        configs.append(config)

    return run_comparison_test("Vector Dimension", configs)


def batch_size_test():
    """Test performance with different batch sizes and request counts."""

    test_configs = [
        (100, 8),    # Small batch
        (500, 16),   # Medium batch
        (1000, 16),  # Large batch
        (2000, 32),  # Very large batch
        (5000, 64),  # Massive batch
    ]

    configs = []
    for requests, workers in test_configs:
        config = BenchmarkConfig(
            total_requests=requests,
            workers=workers,
            query_type="vector_search",
            index_name="sample_index",
            vector_dim=1536,
            num_results=10
        )
        configs.append(config)

    return run_comparison_test("Batch Size", configs)


def results_count_test():
    """Test how the number of returned results affects performance."""

    result_counts = [1, 5, 10, 25, 50, 100]
    configs = []

    for num_results in result_counts:
        config = BenchmarkConfig(
            total_requests=500,
            workers=16,
            query_type="vector_search",
            index_name="sample_index",
            vector_dim=1536,
            num_results=num_results
        )
        configs.append(config)

    return run_comparison_test("Results Count", configs)


def print_summary_table(all_results: dict):
    """Print a summary table of all test results."""

    print(f"\n{'='*80}")
    print("PERFORMANCE COMPARISON SUMMARY")
    print(f"{'='*80}")

    # Create a summary table
    print(f"{'Test':<20} {'Config':<25} {'QPS':<10} {'Avg Lat':<10} {'P95 Lat':<10} {'Success%':<10}")
    print("-" * 80)

    for test_name, results in all_results.items():
        for result in results:
            if 'error' not in result:
                config_summary = f"{result['config']['workers']}w/{result['config']['total_requests']}r"
                print(f"{test_name[:19]:<20} {config_summary:<25} "
                      f"{result['qps']:<10.1f} {result['avg_latency_ms']:<10.1f} "
                      f"{result['p95_latency_ms']:<10.1f} {result['success_rate']:<10.1f}")

    print("\nLegend: w=workers, r=requests, Lat=Latency(ms)")


def main():
    """Run all performance comparison tests."""

    print("Redis Performance Comparison Suite")
    print("This will run multiple benchmark tests to compare performance")
    print("across different configurations.\n")

    # Check if we should proceed
    response = input("This will take several minutes. Continue? (y/N): ")
    if response.lower() != 'y':
        print("Cancelled.")
        return

    all_results = {}

    try:
        # Run all tests
        print("Starting performance comparison tests...")

        all_results["Worker Scaling"] = worker_scaling_test()
        time.sleep(2)  # Brief pause between tests

        all_results["Query Types"] = query_type_comparison()
        time.sleep(2)

        all_results["Vector Dimensions"] = vector_dimension_test()
        time.sleep(2)

        all_results["Batch Sizes"] = batch_size_test()
        time.sleep(2)

        all_results["Result Counts"] = results_count_test()

        # Print final summary
        print_summary_table(all_results)

        print(f"\n{'='*60}")
        print("PERFORMANCE COMPARISON COMPLETE")
        print(f"{'='*60}")
        print("Check the 'results/' directory for detailed JSON reports.")
        print("Use these results to optimize your Redis search configuration!")

    except KeyboardInterrupt:
        print("\n\nTest interrupted by user.")
    except Exception as e:
        print(f"\nError during testing: {e}")
        print("Check your Redis connection and index configuration.")


if __name__ == "__main__":
    main()