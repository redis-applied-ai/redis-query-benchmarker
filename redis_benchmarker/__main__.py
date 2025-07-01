"""Command-line interface for Redis Query Benchmarker."""

import click
import sys
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
@click.option('--ssl', is_flag=True, help='Use SSL connection')
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
@click.option('--config-file', default=None, help='Load configuration from JSON file')
@click.option('--save-config', default=None, help='Save configuration to JSON file')
@click.option('--filter-expression', default=None, help='Filter expression for hybrid search')
@click.option('--search-query', default='*', help='Search query for redis-py search')
@click.option('--max-connections', default=None, type=int, help='Max connections in pool')
@click.option('--qps', default=None, type=float, help='Target queries per second (QPS)')
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
                ssl=kwargs['ssl'],
                max_connections=kwargs['max_connections']
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
                qps=kwargs['qps'],
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

        # Run benchmark
        click.echo("Redis Query Benchmarker")
        click.echo("=" * 40)
        click.echo(f"Target: {config.redis.host}:{config.redis.port}")
        click.echo(f"Query Type: {config.query_type}")
        click.echo(f"Requests: {config.total_requests}")
        click.echo(f"Workers: {config.workers}")
        if config.index_name:
            click.echo(f"Index: {config.index_name}")
        click.echo()

        benchmarker = RedisBenchmarker(config)
        results = benchmarker.run_benchmark()

        # Output results
        if config.output_format == 'console':
            benchmarker.print_results(results)

        if config.output_file:
            benchmarker.save_results(results, config.output_file, config.output_format)

        # Exit with error code if there were failures
        if results.failed_requests > 0:
            click.echo(f"\nWarning: {results.failed_requests} requests failed", err=True)
            if results.success_rate < 95:
                sys.exit(1)

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