"""Configuration management for Redis benchmarker."""

from typing import Optional, Dict, Any
from pydantic import BaseModel, Field, field_validator
import json


class RedisConnectionConfig(BaseModel):
    """Redis connection configuration."""
    host: str = Field(default="localhost", description="Redis host")
    port: int = Field(default=6379, description="Redis port", ge=1, le=65535)
    password: Optional[str] = Field(default=None, description="Redis password")
    username: Optional[str] = Field(default=None, description="Redis username")
    db: int = Field(default=0, description="Redis database number", ge=0)
    tls: bool = Field(default=False, description="Use TLS/SSL connection")
    tls_insecure: bool = Field(default=False, description="Skip TLS certificate validation (allows self-signed certificates)")
    socket_timeout: float = Field(default=30.0, description="Socket timeout in seconds")
    socket_connect_timeout: float = Field(default=30.0, description="Socket connect timeout")
    max_connections: Optional[int] = Field(default=None, description="Max connections in pool")
    protocol: int = Field(default=2, description="Redis protocol version (2 or 3)", ge=2, le=3)


class BenchmarkConfig(BaseModel):
    """Main benchmark configuration."""

    # Connection settings
    redis: RedisConnectionConfig = Field(default_factory=RedisConnectionConfig)

    # Benchmark settings
    total_requests: int = Field(default=1000, description="Total number of requests", ge=1)
    workers: int = Field(default=16, description="Number of worker threads", ge=1, le=256)
    query_type: str = Field(default="vector_search", description="Type of query to execute")

    # Query-specific settings
    index_name: Optional[str] = Field(default=None, description="Redis index name")
    vector_field: Optional[str] = Field(default=None, description="Vector field name")
    vector_dim: int = Field(default=1536, description="Vector dimension", ge=1)
    num_results: int = Field(default=10, description="Number of results to return", ge=1)

    # Output settings
    output_format: str = Field(default="console", description="Output format")
    output_file: Optional[str] = Field(default=None, description="Output file path")
    verbose: bool = Field(default=False, description="Verbose output")
    show_expanded_metrics: bool = Field(default=False, description="Show expanded metrics including result counts and normalized latency")

    # Advanced settings
    warmup_requests: int = Field(default=0, description="Number of warmup requests", ge=0)
    timeout: float = Field(default=30.0, description="Query timeout in seconds")
    qps: Optional[float] = Field(default=None, description="Target queries per second (QPS)")
    sample_file: Optional[str] = Field(default=None, description="File path containing sample queries (one per line)")
    extra_params: Dict[str, Any] = Field(default_factory=dict, description="Extra parameters")

    @field_validator('workers')
    @classmethod
    def validate_workers(cls, v):
        if v > 128:
            print(f"Warning: Using {v} workers may cause resource exhaustion")
        return v

    @field_validator('output_format')
    @classmethod
    def validate_output_format(cls, v):
        if v not in ['console', 'json', 'csv']:
            raise ValueError('Output format must be console, json, or csv')
        return v

    @property
    def max_connections(self) -> int:
        """Calculate appropriate max connections for the connection pool."""
        if self.redis.max_connections:
            return self.redis.max_connections
        return self.workers * 2 + 10  # Leave some buffer

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return self.model_dump()

    def to_json(self) -> str:
        """Convert config to JSON string."""
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_file(cls, file_path: str) -> "BenchmarkConfig":
        """Load configuration from JSON file."""
        with open(file_path, 'r') as f:
            data = json.load(f)
        return cls(**data)

    def save_to_file(self, file_path: str):
        """Save configuration to JSON file."""
        with open(file_path, 'w') as f:
            f.write(self.to_json())