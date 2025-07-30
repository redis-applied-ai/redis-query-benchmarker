"""Command-line interface for Redis Query Benchmarker."""

import click
import sys
import os
from typing import Dict, Any

from .config import BenchmarkConfig, RedisConnectionConfig
from .benchmarker import RedisBenchmarker
from .executors import list_query_executors


@click.command()
@click.option('--host', default='localhost', help='Redis host')
@click.option('--port', default=6379, help='Redis port')
@click.option('--password', default=None, help='Redis password')
@click.option('--username', default=None, help='Redis username')
@click.option('--db', default=0, help='Redis database number')
@click.option('--tls', is_flag=True, help='Use TLS connection')
@click.option('--insecure', is_flag=True, help='Skip TLS certificate validation (allows self-signed certificates)')
@click.option('--total-requests', default=1000, help='Total number of requests')
@click.option('--workers', default=16, help='Number of worker threads')
@click.option('--query-type', default='vector_search',
              help='Type of query to execute (use list-executors to see available types)')
@click.option('--index-name', default=None, help='Redis index name')
@click.option('--vector-field', default='embedding', help='Vector field name')
@click.option('--vector-dim', default=1536, help='Vector dimension')
@click.option('--num-results', default=10, help='Number of results to return')
@click.option('--output-format', default='console',
              type=click.Choice(['console', 'json', 'csv']),
              help='Output format')
@click.option('--output-file', default=None, help='Output file path')
@click.option('--warmup-requests', default=0, help='Number of warmup requests')
@click.option('--timeout', default=30.0, help='Query timeout in seconds')
@click.option('--verbose', is_flag=True, help='Verbose output')
@click.option('--show-expanded-metrics', is_flag=True, help='Show expanded metrics including result counts and normalized latency')
@click.option('--config-file', default=None, help='Load configuration from JSON file')
@click.option('--save-config', default=None, help='Save configuration to JSON file')
@click.option('--filter-expression', default=None, help='Filter expression for hybrid search')
@click.option('--search-query', default='*', help='Search query for redis-py search')
@click.option('--max-connections', default=None, type=int, help='Max connections in pool')
@click.option('--qps', default=None, type=float, help='Target queries per second (QPS)')
@click.option('--sample-file', default=None, help='File path containing sample queries (one per line)')
@click.option('--resp', default=2, type=click.IntRange(2, 3), help='Redis protocol version (2 or 3)')
@click.option('--pre-warm', default=0, type=int, help='Number of connections to pre-warm in the pool (0 = disabled)')
@click.option('--jitter-enabled', is_flag=True, help='Enable QPS jitter for realistic traffic simulation')
@click.option('--jitter-percent', default=10.0, type=float, help='QPS jitter percentage (±)', show_default=True)
@click.option('--jitter-interval-secs', default=5.0, type=float, help='Jitter recalculation interval in seconds', show_default=True)
@click.option('--jitter-distribution', default='uniform', 
              type=click.Choice(['uniform', 'normal', 'triangular', 'bursty']),
              help='Jitter distribution type', show_default=True)
def main(**kwargs):
    """Redis Query Benchmarker - Benchmark Redis search queries with configurable executors."""

    try:
        # Load from config file if provided
        if kwargs['config_file']:
            config = BenchmarkConfig.from_file(kwargs['config_file'])
            click.echo(f"Loaded configuration from {kwargs['config_file']}")
        else:
            # Build config from CLI arguments
            redis_config = RedisConnectionConfig(
                host=kwargs['host'],
                port=kwargs['port'],
                password=kwargs['password'],
                username=kwargs['username'],
                db=kwargs['db'],
                tls=kwargs['tls'],
                tls_insecure=kwargs['insecure'],
                max_connections=kwargs['max_connections'],
                protocol=kwargs['resp']
            )

            # Build extra params for different query types
            extra_params = {}
            if kwargs['filter_expression']:
                extra_params['filter_expression'] = kwargs['filter_expression']
            if kwargs['search_query'] != '*':
                extra_params['search_query'] = kwargs['search_query']

            config = BenchmarkConfig(
                redis=redis_config,
                total_requests=kwargs['total_requests'],
                workers=kwargs['workers'],
                query_type=kwargs['query_type'],
                index_name=kwargs['index_name'],
                vector_field=kwargs['vector_field'],
                vector_dim=kwargs['vector_dim'],
                num_results=kwargs['num_results'],
                output_format=kwargs['output_format'],
                output_file=kwargs['output_file'],
                warmup_requests=kwargs['warmup_requests'],
                timeout=kwargs['timeout'],
                verbose=kwargs['verbose'],
                show_expanded_metrics=kwargs['show_expanded_metrics'],
                qps=kwargs['qps'],
                sample_file=kwargs['sample_file'],
                pre_warm_connections=kwargs['pre_warm'],
                jitter_enabled=kwargs['jitter_enabled'],
                jitter_percent=kwargs['jitter_percent'],
                jitter_interval_secs=kwargs['jitter_interval_secs'],
                jitter_distribution=kwargs['jitter_distribution'],
                extra_params=extra_params
            )

        # Save config if requested
        if kwargs['save_config']:
            config.save_to_file(kwargs['save_config'])
            click.echo(f"Configuration saved to {kwargs['save_config']}")
            return

        # Validate configuration
        if config.query_type in ['vector_search', 'hybrid_search'] and not config.index_name:
            click.echo("Error: --index-name is required for vector and hybrid search queries", err=True)
            sys.exit(1)

        # Validate sample file if provided
        if config.sample_file and not os.path.isfile(config.sample_file):
            click.echo(f"Error: Sample file not found: {config.sample_file}", err=True)
            sys.exit(1)

        # Run benchmark
        click.echo("Redis Query Benchmarker")
        click.echo("=" * 40)
        click.echo(f"Target: {config.redis.host}:{config.redis.port}")
        click.echo(f"Protocol: RESP{config.redis.protocol}")
        click.echo(f"Query Type: {config.query_type}")
        click.echo(f"Requests: {config.total_requests}")
        click.echo(f"Workers: {config.workers}")
        click.echo(f"QPS: {config.qps}")
        if config.qps and config.jitter_enabled:
            click.echo(f"Jitter: ±{config.jitter_percent}% ({config.jitter_distribution}, {config.jitter_interval_secs}s intervals)")
        if config.index_name:
            click.echo(f"Index: {config.index_name}")
        if config.sample_file:
            click.echo(f"Sample File: {config.sample_file}")
        click.echo()

        benchmarker = RedisBenchmarker(config)
        try:
            results = benchmarker.run_benchmark()
        except KeyboardInterrupt:
            # This shouldn't happen since run_benchmark handles KeyboardInterrupt
            # But if it does, we still want to exit gracefully
            click.echo("\nBenchmark interrupted by user.", err=True)
            os._exit(1)

        # Output results (including partial results if interrupted)
        if config.output_format == 'console':
            benchmarker.print_results(results)

        if config.output_file:
            benchmarker.save_results(results, config.output_file, config.output_format)

        # Exit with appropriate code
        if results.metadata.get("interrupted", False):
            click.echo(f"\nBenchmark was interrupted. Partial results shown above.", err=True)
            os._exit(1)
        elif results.failed_requests > 0:
            click.echo(f"\nWarning: {results.failed_requests} requests failed", err=True)
            if results.success_rate < 95:
                sys.exit(1)

    except KeyboardInterrupt:
        click.echo("\nBenchmark interrupted by user.", err=True)
        os._exit(1)
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        if kwargs.get('verbose'):
            import traceback
            traceback.print_exc()
        sys.exit(1)


@click.group()
def cli():
    """Redis Query Benchmarker CLI."""
    pass


@cli.command()
def list_executors():
    """List available query executors."""
    click.echo("Available query executors:")
    for executor in list_query_executors():
        click.echo(f"  - {executor}")


if __name__ == "__main__":
    main()