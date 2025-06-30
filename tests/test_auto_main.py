"""Tests for auto-discovery and auto-main functionality."""

import pytest
import sys
from unittest.mock import patch, MagicMock
from redis_benchmarker.executors import (
    BaseQueryExecutor,
    _discover_executors_in_module,
    enable_auto_main
)


class TestExecutorNaming:
    """Test executor name generation."""

    def test_default_naming_conversion(self):
        """Test that class names are converted to snake_case correctly."""

        class MyCustomExecutor(BaseQueryExecutor):
            def execute_query(self, redis_client):
                pass

        assert MyCustomExecutor.get_executor_name() == "my_custom"

    def test_executor_suffix_removal(self):
        """Test that 'Executor' suffix is removed."""

        class VectorSearchExecutor(BaseQueryExecutor):
            def execute_query(self, redis_client):
                pass

        assert VectorSearchExecutor.get_executor_name() == "vector_search"

    def test_custom_executor_name(self):
        """Test that custom executor_name attribute is respected."""

        class CustomExecutor(BaseQueryExecutor):
            executor_name = "my_special_name"

            def execute_query(self, redis_client):
                pass

        assert CustomExecutor.get_executor_name() == "my_special_name"

    def test_single_word_class(self):
        """Test single word class names."""

        class SearchExecutor(BaseQueryExecutor):
            def execute_query(self, redis_client):
                pass

        assert SearchExecutor.get_executor_name() == "search"

    def test_acronym_handling(self):
        """Test handling of acronyms in class names."""

        class AIMLExecutor(BaseQueryExecutor):
            def execute_query(self, redis_client):
                pass

        assert AIMLExecutor.get_executor_name() == "a_i_m_l"


class TestModuleDiscovery:
    """Test module discovery functionality."""

    def test_discover_executors_empty_module(self):
        """Test discovery in module with no executors."""
        # Create a mock module with no BaseQueryExecutor subclasses
        mock_module = MagicMock()
        mock_module.__dict__ = {"SomeClass": str, "another_var": 42}

        with patch.dict(sys.modules, {"test_module": mock_module}):
            with patch('inspect.getmembers', return_value=[]):
                executors = _discover_executors_in_module("test_module")

        assert executors == []

    def test_discover_executors_with_executors(self):
        """Test discovery in module with executors."""

        class TestExecutor1(BaseQueryExecutor):
            def execute_query(self, redis_client):
                pass

        class TestExecutor2(BaseQueryExecutor):
            def execute_query(self, redis_client):
                pass

        class NotAnExecutor:
            pass

        mock_module = MagicMock()
        mock_module.__name__ = "test_module"

        with patch.dict(sys.modules, {"test_module": mock_module}):
            with patch('inspect.getmembers') as mock_getmembers:
                mock_getmembers.return_value = [
                    ("TestExecutor1", TestExecutor1),
                    ("TestExecutor2", TestExecutor2),
                    ("NotAnExecutor", NotAnExecutor),
                    ("BaseQueryExecutor", BaseQueryExecutor)
                ]

                # Set the module attribute for the test classes
                TestExecutor1.__module__ = "test_module"
                TestExecutor2.__module__ = "test_module"
                NotAnExecutor.__module__ = "test_module"

                executors = _discover_executors_in_module("test_module")

        assert len(executors) == 2
        assert TestExecutor1 in executors
        assert TestExecutor2 in executors
        assert BaseQueryExecutor not in executors
        assert NotAnExecutor not in executors


class TestAutoMain:
    """Test enable_auto_main functionality."""

    def test_enable_auto_main_not_main(self):
        """Test that enable_auto_main does nothing when module is not __main__."""
        # Should do nothing and not raise any exceptions
        enable_auto_main("some_other_module")

    @patch('redis_benchmarker.executors._discover_executors_in_module')
    @patch('sys.argv', ['script.py'])
    def test_enable_auto_main_no_executors(self, mock_discover):
        """Test behavior when no executors are found."""
        mock_discover.return_value = []

        with pytest.raises(SystemExit):
            enable_auto_main("__main__")

    @patch('redis_benchmarker.executors._discover_executors_in_module')
    @patch('redis_benchmarker.executors.register_query_executor')
    @patch('sys.argv', ['script.py'])
    def test_enable_auto_main_single_executor(self, mock_register, mock_discover):
        """Test behavior with single executor (should auto-add query-type)."""

        class TestExecutor(BaseQueryExecutor):
            def execute_query(self, redis_client):
                pass

        mock_discover.return_value = [TestExecutor]

        with patch('redis_benchmarker.__main__.main') as mock_cli:
            mock_cli.side_effect = SystemExit(0)

            with pytest.raises(SystemExit):
                enable_auto_main("__main__")

        # Should register the executor
        mock_register.assert_called_once_with("test", TestExecutor)

        # Should add query-type to sys.argv
        assert "--query-type" in sys.argv
        assert "test" in sys.argv

    @patch('redis_benchmarker.executors._discover_executors_in_module')
    @patch('redis_benchmarker.executors.register_query_executor')
    @patch('sys.argv', ['script.py', '--query-type', 'custom'])
    def test_enable_auto_main_multiple_executors_with_query_type(self, mock_register, mock_discover):
        """Test behavior with multiple executors when query-type is specified."""

        class TestExecutor1(BaseQueryExecutor):
            def execute_query(self, redis_client):
                pass

        class TestExecutor2(BaseQueryExecutor):
            def execute_query(self, redis_client):
                pass

        mock_discover.return_value = [TestExecutor1, TestExecutor2]

        with patch('redis_benchmarker.__main__.main') as mock_cli:
            mock_cli.side_effect = SystemExit(0)

            with pytest.raises(SystemExit):
                enable_auto_main("__main__")

        # Should register both executors
        assert mock_register.call_count == 2

    @patch('redis_benchmarker.executors._discover_executors_in_module')
    @patch('sys.argv', ['script.py'])
    def test_enable_auto_main_multiple_executors_no_query_type(self, mock_discover):
        """Test behavior with multiple executors when no query-type is specified."""

        class TestExecutor1(BaseQueryExecutor):
            def execute_query(self, redis_client):
                pass

        class TestExecutor2(BaseQueryExecutor):
            def execute_query(self, redis_client):
                pass

        mock_discover.return_value = [TestExecutor1, TestExecutor2]

        with pytest.raises(SystemExit):
            enable_auto_main("__main__")

    @patch('redis_benchmarker.executors._discover_executors_in_module')
    @patch('redis_benchmarker.executors.register_query_executor')
    @patch('sys.argv', ['script.py'])
    def test_enable_auto_main_with_default_executor_name(self, mock_register, mock_discover):
        """Test behavior with default_executor_name parameter."""

        class TestExecutor1(BaseQueryExecutor):
            executor_name = "executor1"
            def execute_query(self, redis_client):
                pass

        class TestExecutor2(BaseQueryExecutor):
            executor_name = "executor2"
            def execute_query(self, redis_client):
                pass

        mock_discover.return_value = [TestExecutor1, TestExecutor2]

        with patch('redis_benchmarker.__main__.main') as mock_cli:
            mock_cli.side_effect = SystemExit(0)

            with pytest.raises(SystemExit):
                enable_auto_main("__main__", default_executor_name="executor1")

        # Should add the specified default executor to sys.argv
        assert "--query-type" in sys.argv
        assert "executor1" in sys.argv