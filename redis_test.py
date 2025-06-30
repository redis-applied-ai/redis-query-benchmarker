import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import redis
from redisvl.index import SearchIndex
from redisvl.query import VectorQuery

# Import the timing utility
import sys
sys.path.insert(0, '.')  # Add current directory to path
from redis_benchmarker.utils import time_operation

workers = 16
# Create a connection pool
pool = redis.ConnectionPool(
    host="localhost",
    port=6379,
    # port=16347,
    # decode_responses=False,  # Important: set to False to handle binary data
    # username="default",
    # password="PASSWORD",
    max_connections=workers*2
)

schema_r = redis.Redis(connection_pool=pool)
schema = SearchIndex.from_existing("idx_2a34cb12eab8515e", schema_r).schema
schema_r.close()

def make_query():
    """Execute a single query and return result with timing"""
    # Create a new Redis client for each query
    r = redis.Redis(connection_pool=pool)

    try:
        # Create a new index instance for each query
        index = SearchIndex(schema, r)
        vquery = VectorQuery(
            vector=list(np.array([random.random() for _ in range(2545)], dtype=np.float16).tolist()),
            vector_field_name="2a34cb12eab8515e",
            num_results=10,
            return_score=True,
            dtype="float16",
            hybrid_policy="BATCHES",
            batch_size=250,
            #hybrid_policy="ADHOC_BF",
            filter_expression="(@__schema__:{ProductSchema} @__schema_field__ProductSchema_inventory__is_active:{1})",
        )

        with time_operation() as latency_ms:
            res = index.query(vquery)

        return {"result": res, "latency_ms": float(latency_ms)}
    finally:
        # Explicitly release the connection back to the pool
        r.close()


# Parameters
total_requests = 100


def stress_test():
    start_time = time.time()
    print(f"total_requests: {total_requests}")
    results = []
    latencies = []

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(make_query) for _ in range(total_requests)]

        for future in as_completed(futures):
            try:
                query_data = future.result()
                results.append(query_data["result"])
                latencies.append(query_data["latency_ms"])
            except Exception as e:
                print(f"Error during query: {e}")

    end_time = time.time()
    total_time = end_time - start_time

    # Calculate QPS
    successful_requests = len(latencies)
    qps = successful_requests / total_time

    # Calculate latency statistics
    if latencies:
        avg_latency = np.mean(latencies)
        median_latency = np.median(latencies)
        min_latency = np.min(latencies)
        max_latency = np.max(latencies)
        p90_latency = np.percentile(latencies, 90)
        p95_latency = np.percentile(latencies, 95)
        p99_latency = np.percentile(latencies, 99)
        std_latency = np.std(latencies)

    print(f"\nSent {successful_requests} queries in {total_time:.2f} seconds")
    print(f"Average QPS: {qps:.2f}")

    if latencies:
        print(f"\nLatency Statistics (ms):")
        print(f"  Count:      {len(latencies):,}")
        print(f"  Average:    {avg_latency:.2f}")
        print(f"  Median:     {median_latency:.2f}")
        print(f"  Min:        {min_latency:.2f}")
        print(f"  Max:        {max_latency:.2f}")
        print(f"  Std Dev:    {std_latency:.2f}")
        print(f"  90th pct:   {p90_latency:.2f}")
        print(f"  95th pct:   {p95_latency:.2f}")
        print(f"  99th pct:   {p99_latency:.2f}")

        # Simple latency distribution
        print(f"\nLatency Distribution:")
        bins = [0, 100, 200, 500, 1000, 2000, float("inf")]
        labels = ["<100ms", "100-200ms", "200-500ms", "500ms-1s", "1-2s", ">2s"]

        for i, (bin_max, label) in enumerate(zip(bins[1:], labels)):
            bin_min = bins[i]
            count = sum(1 for lat in latencies if bin_min <= lat < bin_max)
            percentage = (count / len(latencies)) * 100
            print(f"  {label:>10}: {count:4d} ({percentage:5.1f}%)")

    return results, latencies


results, latencies = stress_test()
print(f"\nFirst result sample: {results[0][:3]}...")  # Show first 3 results

# Test basic timing functionality
print("Testing time_operation utility...")

with time_operation() as latency_ms:
    time.sleep(0.1)  # Simulate 100ms operation

print(f"Measured latency: {latency_ms} ms (should be ~100ms)")
assert 90 <= latency_ms <= 150, f"Expected ~100ms, got {latency_ms}ms"
