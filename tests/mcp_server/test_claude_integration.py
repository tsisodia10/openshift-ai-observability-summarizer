"""Tests for Claude Desktop integration functionality."""

import pytest
import json
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from mcp_server.claude_integration import PrometheusChatBot


class TestPrometheusChatBot:
    """Test Claude Desktop-style chatbot functionality."""
    
    @pytest.fixture
    def mock_anthropic_client(self):
        """Mock Anthropic client for testing."""
        with patch('mcp_server.claude_integration.anthropic') as mock_anthropic:
            mock_client = Mock()
            mock_anthropic.Anthropic.return_value = mock_client
            yield mock_client
    
    @pytest.fixture
    def mock_mcp_server(self):
        """Mock MCP server for testing."""
        with patch('mcp_server.claude_integration.ObservabilityMCPServer') as mock_server:
            mock_instance = Mock()
            mock_instance.mcp._tool_manager._tools = {
                'search_metrics': Mock(),
                'execute_promql': Mock(),
                'get_metric_metadata': Mock()
            }
            mock_server.return_value = mock_instance
            yield mock_instance
    
    @pytest.fixture
    def mock_mcp_client(self):
        """Mock MCP client helper for testing."""
        with patch('mcp_server.claude_integration.MCPClientHelper') as mock_client:
            mock_instance = Mock()
            mock_client.return_value = mock_instance
            yield mock_instance
    
    def test_chatbot_initialization_with_api_key(self, mock_anthropic_client, mock_mcp_server, mock_mcp_client):
        """Test chatbot initializes correctly with API key."""
        chatbot = PrometheusChatBot(api_key="test-key", model_name="claude-3-5-haiku-20241022")
        
        assert chatbot.api_key == "test-key"
        assert chatbot.model_name == "claude-3-5-haiku-20241022"
        assert chatbot.claude_client is not None
    
    def test_chatbot_initialization_without_api_key(self, mock_mcp_server, mock_mcp_client):
        """Test chatbot handles missing API key gracefully."""
        with patch.dict(os.environ, {}, clear=True):
            chatbot = PrometheusChatBot()
            assert chatbot.claude_client is None
    
    def test_convert_mcp_tools_to_claude_schema(self, mock_anthropic_client, mock_mcp_server, mock_mcp_client):
        """Test MCP tools are converted to Claude schema format correctly."""
        chatbot = PrometheusChatBot(api_key="test-key")
        claude_tools = chatbot._convert_mcp_tools_to_claude_schema()
        
        assert isinstance(claude_tools, list)
        assert len(claude_tools) > 0
        
        # Check tool schema structure
        for tool in claude_tools:
            assert "name" in tool
            assert "description" in tool
            assert "input_schema" in tool
            assert tool["input_schema"]["type"] == "object"
    
    def test_route_tool_call_to_mcp(self, mock_anthropic_client, mock_mcp_server, mock_mcp_client):
        """Test tool calls are routed to MCP server correctly."""
        chatbot = PrometheusChatBot(api_key="test-key")
        
        # Mock MCP client response
        mock_mcp_client.call_tool_sync.return_value = [{"text": "test result"}]
        
        result = chatbot._route_tool_call_to_mcp("search_metrics", {"pattern": "cpu"})
        
        mock_mcp_client.call_tool_sync.assert_called_once_with("search_metrics", {"pattern": "cpu"})
        assert result == "test result"
    
    def test_chat_without_claude_client(self, mock_mcp_server, mock_mcp_client):
        """Test chat handles missing Claude client gracefully."""
        chatbot = PrometheusChatBot()  # No API key
        
        result = chatbot.chat("test question")
        assert "Claude AI not available" in result
    
    @patch('mcp_server.claude_integration.anthropic')
    def test_chat_with_tool_calling(self, mock_anthropic, mock_mcp_server, mock_mcp_client):
        """Test full chat flow with tool calling."""
        # Setup mocks
        mock_client = Mock()
        mock_anthropic.Anthropic.return_value = mock_client
        
        # Mock Claude response with tool use
        mock_content_block = Mock()
        mock_content_block.type = "tool_use"
        mock_content_block.name = "search_metrics"
        mock_content_block.input = {"pattern": "cpu"}
        mock_content_block.id = "tool_123"
        
        mock_response = Mock()
        mock_response.stop_reason = "tool_use"
        mock_response.content = [mock_content_block]
        
        # Mock final response
        mock_final_content = Mock()
        mock_final_content.type = "text"
        mock_final_content.text = "CPU usage is 50%"
        
        mock_final_response = Mock()
        mock_final_response.stop_reason = "stop"
        mock_final_response.content = [mock_final_content]
        
        mock_client.messages.create.side_effect = [mock_response, mock_final_response]
        
        # Mock MCP tool response
        mock_mcp_client.call_tool_sync.return_value = [{"text": "cpu metrics found"}]
        
        chatbot = PrometheusChatBot(api_key="test-key")
        result = chatbot.chat("What's the CPU usage?")
        
        # Verify tool was called
        mock_mcp_client.call_tool_sync.assert_called_with("search_metrics", {"pattern": "cpu"})
        assert "CPU usage is 50%" in result
    
    def test_test_connection(self, mock_anthropic_client, mock_mcp_server, mock_mcp_client):
        """Test connection testing functionality."""
        chatbot = PrometheusChatBot(api_key="test-key")
        
        # Mock successful connection
        mock_mcp_server.mcp._tool_manager._tools = {"search_metrics": Mock()}
        
        result = chatbot.test_connection()
        assert result is True
    
    def test_system_prompt_generation(self, mock_anthropic_client, mock_mcp_server, mock_mcp_client):
        """Test system prompt is generated correctly."""
        chatbot = PrometheusChatBot(api_key="test-key")
        
        prompt = chatbot._create_claude_desktop_system_prompt("test-namespace")
        
        assert "expert Kubernetes and Prometheus observability assistant" in prompt
        assert "test-namespace" in prompt
        assert "find_best_metric_with_metadata_v2" in prompt
        assert "execute_promql" in prompt
    
    def test_progress_callback_integration(self, mock_anthropic_client, mock_mcp_server, mock_mcp_client):
        """Test progress callback is called during chat."""
        chatbot = PrometheusChatBot(api_key="test-key")
        
        # Mock Claude response without tool use
        mock_content = Mock()
        mock_content.type = "text"
        mock_content.text = "Analysis complete"
        
        mock_response = Mock()
        mock_response.stop_reason = "stop"
        mock_response.content = [mock_content]
        
        mock_anthropic_client.messages.create.return_value = mock_response
        
        # Test with progress callback
        progress_calls = []
        def progress_callback(msg):
            progress_calls.append(msg)
        
        result = chatbot.chat("test question", progress_callback=progress_callback)
        
        assert len(progress_calls) > 0
        assert any("Thinking" in call for call in progress_calls)


class TestClaudeIntegrationHelpers:
    """Test helper functions in Claude integration."""
    
    def test_tool_schema_conversion(self):
        """Test that MCP tools are converted to proper Claude schema."""
        # This would test the _convert_mcp_tools_to_claude_schema method
        # with actual MCP tool definitions
        pass
    
    def test_error_handling(self):
        """Test error handling in tool routing."""
        # This would test error scenarios in _route_tool_call_to_mcp
        pass
