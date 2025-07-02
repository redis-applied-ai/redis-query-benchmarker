"""Tests for configuration module."""

import pytest
import json
import tempfile
import os
from redis_benchmarker.config import BenchmarkConfig, RedisConnectionConfig


class TestRedisConnectionConfig:
    """Test Redis connection configuration."""

    def test_default_values(self):
        """Test default configuration values."""
        config = RedisConnectionConfig()
        assert config.host == "localhost"
        assert config.port == 6379
        assert config.password is None
        assert config.username is None
        assert config.db == 0
        assert config.tls is False

    def test_custom_values(self):
        """Test custom configuration values."""
        config = RedisConnectionConfig(
            host="redis.example.com",
            port=16379,
            password="secret",
            username="user",
            db=1,
            tls=True
        )
        assert config.host == "redis.example.com"
        assert config.port == 16379
        assert config.password == "secret"
        assert config.username == "user"
        assert config.db == 1
        assert config.tls is True

    def test_port_validation(self):
        """Test port validation."""
        with pytest.raises(ValueError):
            RedisConnectionConfig(port=0)

        with pytest.raises(ValueError):
            RedisConnectionConfig(port=65536)


class TestBenchmarkConfig:
    """Test benchmark configuration."""

    def test_default_values(self):
        """Test default configuration values."""
        config = BenchmarkConfig()
        assert config.total_requests == 1000
        assert config.workers == 16
        assert config.query_type == "vector_search"
        assert config.vector_dim == 1536
        assert config.num_results == 10
        assert config.output_format == "console"
        assert config.verbose is False

    def test_max_connections_property(self):
        """Test max_connections property calculation."""
        config = BenchmarkConfig(workers=8)
        assert config.max_connections == 26  # 8 * 2 + 10

        config = BenchmarkConfig(workers=16)
        assert config.max_connections == 42  # 16 * 2 + 10

        # Test with explicit max_connections
        config = BenchmarkConfig(workers=8)
        config.redis.max_connections = 50
        assert config.max_connections == 50

    def test_workers_validation(self):
        """Test workers validation warning."""
        # Should not raise error, but may print warning
        config = BenchmarkConfig(workers=200)
        assert config.workers == 200

    def test_output_format_validation(self):
        """Test output format validation."""
        config = BenchmarkConfig(output_format="json")
        assert config.output_format == "json"

        with pytest.raises(ValueError):
            BenchmarkConfig(output_format="invalid")

    def test_to_dict(self):
        """Test configuration to dictionary conversion."""
        config = BenchmarkConfig(
            total_requests=500,
            workers=8,
            query_type="hybrid_search"
        )
        config_dict = config.to_dict()

        assert config_dict["total_requests"] == 500
        assert config_dict["workers"] == 8
        assert config_dict["query_type"] == "hybrid_search"
        assert "redis" in config_dict

    def test_to_json(self):
        """Test configuration to JSON conversion."""
        config = BenchmarkConfig(total_requests=500)
        json_str = config.to_json()

        # Should be valid JSON
        parsed = json.loads(json_str)
        assert parsed["total_requests"] == 500

    def test_file_operations(self):
        """Test saving and loading configuration from file."""
        config = BenchmarkConfig(
            total_requests=500,
            workers=8,
            query_type="redis_py"
        )

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_file = f.name

        try:
            # Save to file
            config.save_to_file(temp_file)
            assert os.path.exists(temp_file)

            # Load from file
            loaded_config = BenchmarkConfig.from_file(temp_file)
            assert loaded_config.total_requests == 500
            assert loaded_config.workers == 8
            assert loaded_config.query_type == "redis_py"

        finally:
            os.unlink(temp_file)

    def test_qps_field(self):
        """Test qps field in BenchmarkConfig."""
        config = BenchmarkConfig(qps=123.45)
        assert config.qps == 123.45
        config_dict = config.to_dict()
        assert 'qps' in config_dict
        assert config_dict['qps'] == 123.45
        # Test default is None
        config2 = BenchmarkConfig()
        assert config2.qps is None
        # Test serialization/deserialization
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_file = f.name
        try:
            config.save_to_file(temp_file)
            loaded = BenchmarkConfig.from_file(temp_file)
            assert loaded.qps == 123.45
        finally:
            os.unlink(temp_file)