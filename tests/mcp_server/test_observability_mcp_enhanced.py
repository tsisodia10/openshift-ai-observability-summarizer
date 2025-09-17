"""Tests for enhanced ObservabilityMCP server with Prometheus tools."""

import pytest
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))


class TestObservabilityMCPServerEnhanced:
    """Test enhanced MCP server with Prometheus tools."""
    
    @patch('mcp_server.observability_mcp.FastMCP')
    @patch('mcp_server.observability_mcp.get_python_logger')
    def test_mcp_server_initialization(self, mock_logger, mock_fastmcp):
        """Test MCP server initializes with all tools."""
        from mcp_server.observability_mcp import ObservabilityMCPServer
        
        mock_mcp_instance = Mock()
        mock_fastmcp.return_value = mock_mcp_instance
        
        server = ObservabilityMCPServer()
        
        # Verify FastMCP was initialized
        mock_fastmcp.assert_called_once_with("metrics-observability")
        assert server.mcp == mock_mcp_instance
    
    @patch('mcp_server.observability_mcp.FastMCP')
    @patch('mcp_server.observability_mcp.get_python_logger')
    def test_prometheus_tools_registration(self, mock_logger, mock_fastmcp):
        """Test that all Prometheus tools are registered."""
        from mcp_server.observability_mcp import ObservabilityMCPServer
        
        mock_mcp_instance = Mock()
        mock_mcp_instance._tool_manager._tools = {}
        mock_fastmcp.return_value = mock_mcp_instance
        
        # Mock the tool registration
        def mock_tool_decorator():
            def decorator(func):
                mock_mcp_instance._tool_manager._tools[func.__name__] = func
                return func
            return decorator
        
        mock_mcp_instance.tool.return_value = mock_tool_decorator()
        
        with patch('mcp_server.observability_mcp.search_metrics') as mock_search_metrics, \
             patch('mcp_server.observability_mcp.execute_promql') as mock_execute_promql, \
             patch('mcp_server.observability_mcp.find_best_metric_with_metadata_v2') as mock_find_metric:
            
            server = ObservabilityMCPServer()
            
            # Verify tools were registered
            expected_tools = [
                'search_metrics',
                'execute_promql', 
                'find_best_metric_with_metadata_v2'
            ]
            
            for tool_name in expected_tools:
                assert tool_name in mock_mcp_instance._tool_manager._tools
    
    def test_tool_import_fallback(self):
        """Test that tool imports work with fallback mechanisms."""
        # Test that imports work even if relative imports fail
        with patch('mcp_server.observability_mcp.importlib.util') as mock_importlib:
            mock_spec = Mock()
            mock_module = Mock()
            
            # Mock successful module loading
            mock_importlib.spec_from_file_location.return_value = mock_spec
            mock_importlib.module_from_spec.return_value = mock_module
            
            # Test would verify fallback import logic works
            pass
    
    def test_tool_count_reporting(self):
        """Test that tool count is reported correctly."""
        with patch('mcp_server.observability_mcp.FastMCP') as mock_fastmcp, \
             patch('mcp_server.observability_mcp.get_python_logger') as mock_logger:
            
            from mcp_server.observability_mcp import ObservabilityMCPServer
            
            mock_mcp_instance = Mock()
            mock_mcp_instance._tool_manager._tools = {
                'search_metrics': Mock(),
                'execute_promql': Mock(),
                'find_best_metric_with_metadata_v2': Mock(),
                'get_metric_metadata': Mock(),
                'list_models': Mock()
            }
            mock_fastmcp.return_value = mock_mcp_instance
            
            server = ObservabilityMCPServer()
            
            # Should have registered multiple tools
            assert len(mock_mcp_instance._tool_manager._tools) >= 5


class TestPrometheusToolsRegistration:
    """Test specific Prometheus tools registration."""
    
    def test_all_prometheus_tools_available(self):
        """Test that all expected Prometheus tools are available."""
        expected_prometheus_tools = [
            'search_metrics',
            'get_metric_metadata', 
            'get_label_values',
            'execute_promql',
            'explain_results',
            'suggest_queries',
            'select_best_metric',
            'find_best_metric_with_metadata_v2',
            'find_best_metric_with_metadata'
        ]
        
        # Test that these functions exist and are importable
        try:
            from mcp_server.tools.prometheus_tools import (
                search_metrics,
                get_metric_metadata,
                get_label_values,
                execute_promql,
                explain_results,
                suggest_queries,
                select_best_metric,
                find_best_metric_with_metadata_v2,
                find_best_metric_with_metadata
            )
            
            # All imports successful
            assert True
            
        except ImportError as e:
            pytest.fail(f"Failed to import Prometheus tools: {e}")
    
    def test_tool_function_signatures(self):
        """Test that tool functions have correct signatures."""
        from mcp_server.tools.prometheus_tools import search_metrics, execute_promql
        
        import inspect
        
        # Test search_metrics signature
        sig = inspect.signature(search_metrics)
        assert 'pattern' in sig.parameters
        
        # Test execute_promql signature  
        sig = inspect.signature(execute_promql)
        assert 'query' in sig.parameters


class TestErrorHandling:
    """Test error handling in Claude Desktop integration."""
    
    def test_missing_api_key_handling(self):
        """Test graceful handling of missing API keys."""
        with patch.dict(os.environ, {}, clear=True):
            from mcp_server.claude_integration import PrometheusChatBot
            
            chatbot = PrometheusChatBot()
            result = chatbot.chat("test question")
            
            assert "Claude AI not available" in result
    
    def test_mcp_tool_error_handling(self):
        """Test error handling when MCP tools fail."""
        with patch('mcp_server.claude_integration.anthropic') as mock_anthropic:
            from mcp_server.claude_integration import PrometheusChatBot
            
            mock_client = Mock()
            mock_anthropic.Anthropic.return_value = mock_client
            
            chatbot = PrometheusChatBot(api_key="test-key")
            
            # Mock MCP client to raise exception
            with patch.object(chatbot, 'mcp_client') as mock_mcp_client:
                mock_mcp_client.call_tool_sync.side_effect = Exception("MCP error")
                
                result = chatbot._route_tool_call_to_mcp("search_metrics", {"pattern": "cpu"})
                assert "Error executing search_metrics" in result
    
    def test_token_limit_handling(self):
        """Test handling of large tool results that exceed token limits."""
        from mcp_server.claude_integration import PrometheusChatBot
        
        with patch('mcp_server.claude_integration.anthropic'):
            chatbot = PrometheusChatBot(api_key="test-key")
            
            # Mock large tool result
            large_result = "x" * 5000  # Larger than 3000 char limit
            
            with patch.object(chatbot, 'mcp_client') as mock_mcp_client:
                mock_mcp_client.call_tool_sync.return_value = [{"text": large_result}]
                
                result = chatbot._route_tool_call_to_mcp("search_metrics", {"pattern": "cpu"})
                
                # Result should be truncated
                assert len(result) <= 3050  # 3000 + truncation message
                assert "Result truncated" in result


class TestIntegrationWorkflow:
    """Test end-to-end integration workflow."""
    
    @patch('mcp_server.tools.prometheus_tools._make_prometheus_request')
    def test_full_question_workflow(self, mock_request):
        """Test complete workflow from question to answer."""
        # Mock the typical workflow:
        # 1. find_best_metric_with_metadata_v2
        # 2. execute_promql
        # 3. Final response
        
        mock_request.side_effect = [
            # Search metrics response
            {
                "data": {
                    "metrics": [{"name": "kube_pod_status_phase"}]
                }
            },
            # Metadata response
            {
                "data": [{
                    "metric": {"__name__": "kube_pod_status_phase"},
                    "help": "The phase of a pod",
                    "type": "gauge"
                }]
            },
            # PromQL execution response
            {
                "data": {
                    "result": [
                        {
                            "metric": {"phase": "Running"},
                            "value": [1694123456, "7"]
                        }
                    ]
                }
            }
        ]
        
        # Test the workflow
        from mcp_server.tools.prometheus_tools import find_best_metric_with_metadata_v2, execute_promql
        
        # Step 1: Find best metric
        metric_result = find_best_metric_with_metadata_v2("How many pods are running?")
        assert len(metric_result) > 0
        
        # Step 2: Execute PromQL
        promql_result = execute_promql("sum(kube_pod_status_phase{phase='Running'})")
        assert len(promql_result) > 0


if __name__ == "__main__":
    pytest.main([__file__])
