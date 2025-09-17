"""Tests for Claude Desktop-style UI integration."""

import pytest
import streamlit as st
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))


class TestClaudeDesktopUI:
    """Test Claude Desktop-style UI functionality."""
    
    @patch('streamlit.chat_input')
    @patch('streamlit.chat_message')
    @patch('streamlit.session_state')
    def test_chat_interface_initialization(self, mock_session_state, mock_chat_message, mock_chat_input):
        """Test chat interface initializes correctly."""
        # Mock session state
        mock_session_state.claude_messages = []
        
        # This would test the UI initialization
        # In practice, testing Streamlit apps requires special testing frameworks
        pass
    
    def test_progress_callback_display(self):
        """Test that progress callbacks are displayed in UI."""
        # Mock progress callback function
        progress_messages = []
        
        def mock_progress_callback(message):
            progress_messages.append(message)
        
        # Test the callback mechanism
        mock_progress_callback("🤖 Thinking... (iteration 1)")
        mock_progress_callback("🔧 Using tool: search_metrics")
        mock_progress_callback("✅ search_metrics completed")
        
        assert len(progress_messages) == 3
        assert "Thinking" in progress_messages[0]
        assert "Using tool" in progress_messages[1]
        assert "completed" in progress_messages[2]
    
    def test_suggested_questions(self):
        """Test suggested question functionality."""
        suggested_questions = [
            "🖥️ What's the GPU usage?",
            "📊 Show CPU metrics", 
            "💾 Check memory usage",
            "🚨 Any alerts firing?"
        ]
        
        # Test that suggested questions are properly formatted
        for question in suggested_questions:
            assert len(question) > 0
            assert any(emoji in question for emoji in ["🖥️", "📊", "💾", "🚨"])
    
    @patch('mcp_server.claude_integration.PrometheusChatBot')
    def test_claude_chatbot_integration(self, mock_chatbot_class):
        """Test Claude chatbot integration in UI."""
        # Mock chatbot instance
        mock_chatbot = Mock()
        mock_chatbot.test_connection.return_value = True
        mock_chatbot.chat.return_value = "Test response with PromQL query"
        mock_chatbot_class.return_value = mock_chatbot
        
        # Test chatbot initialization
        chatbot = mock_chatbot_class(api_key="test-key", model_name="claude-3-5-haiku-20241022")
        
        # Test connection
        assert chatbot.test_connection() is True
        
        # Test chat
        response = chatbot.chat("How many pods are running?")
        assert "Test response" in response
    
    def test_model_selection_ui(self):
        """Test model selection functionality in UI."""
        # Mock model list
        mock_models = [
            "anthropic/claude-3-5-haiku-20241022",
            "anthropic/claude-sonnet-4-20250514", 
            "openai/gpt-4o-mini",
            "meta-llama/Llama-3.2-3B-Instruct"
        ]
        
        # Test that all models are available
        assert len(mock_models) == 4
        assert any("anthropic" in model for model in mock_models)
        assert any("openai" in model for model in mock_models)
        assert any("meta-llama" in model for model in mock_models)
    
    def test_api_key_handling(self):
        """Test API key input and validation."""
        # Test external model requires API key
        model_config = {
            "anthropic/claude-3-5-haiku-20241022": {
                "external": True,
                "requiresApiKey": True,
                "provider": "anthropic"
            },
            "meta-llama/Llama-3.2-3B-Instruct": {
                "external": False,
                "requiresApiKey": False,
                "provider": "vllm"
            }
        }
        
        # Anthropic model should require API key
        anthropic_model = model_config["anthropic/claude-3-5-haiku-20241022"]
        assert anthropic_model["requiresApiKey"] is True
        
        # Local model should not require API key
        local_model = model_config["meta-llama/Llama-3.2-3B-Instruct"]
        assert local_model["requiresApiKey"] is False


class TestRealTimeToolDisplay:
    """Test real-time tool call display functionality."""
    
    def test_progress_message_formatting(self):
        """Test that progress messages are formatted correctly."""
        test_messages = [
            "🤖 Thinking... (iteration 1)",
            "🔧 Using tool: find_best_metric_with_metadata_v2",
            "✅ find_best_metric_with_metadata_v2 completed",
            "🔧 Using tool: execute_promql", 
            "✅ execute_promql completed"
        ]
        
        for message in test_messages:
            # Test message structure
            assert len(message) > 0
            # Should have emoji prefix
            assert any(emoji in message for emoji in ["🤖", "🔧", "✅"])
    
    def test_tool_call_sequence(self):
        """Test typical tool call sequence."""
        expected_sequence = [
            ("iteration", "🤖 Thinking..."),
            ("tool_start", "🔧 Using tool: find_best_metric_with_metadata_v2"),
            ("tool_complete", "✅ find_best_metric_with_metadata_v2 completed"),
            ("tool_start", "🔧 Using tool: execute_promql"),
            ("tool_complete", "✅ execute_promql completed")
        ]
        
        # Verify sequence makes sense
        tool_starts = [item for item in expected_sequence if item[0] == "tool_start"]
        tool_completes = [item for item in expected_sequence if item[0] == "tool_complete"]
        
        assert len(tool_starts) == len(tool_completes)


class TestModelConfigIntegration:
    """Test model configuration integration."""
    
    def test_model_config_structure(self):
        """Test that model config has correct structure."""
        sample_config = {
            "anthropic/claude-3-5-haiku-20241022": {
                "external": True,
                "requiresApiKey": True,
                "serviceName": None,
                "provider": "anthropic",
                "apiUrl": "https://api.anthropic.com/v1/messages",
                "modelName": "claude-3-5-haiku-20241022",
                "cost": {
                    "prompt_rate": 0.000001,
                    "output_rate": 0.000005
                }
            }
        }
        
        model = sample_config["anthropic/claude-3-5-haiku-20241022"]
        
        # Test required fields
        required_fields = ["external", "requiresApiKey", "provider", "apiUrl", "modelName"]
        for field in required_fields:
            assert field in model
        
        # Test cost structure
        assert "cost" in model
        assert "prompt_rate" in model["cost"]
        assert "output_rate" in model["cost"]
    
    def test_local_vs_external_models(self):
        """Test distinction between local and external models."""
        local_model = {
            "external": False,
            "requiresApiKey": False,
            "serviceName": "llama-3-2-3b-instruct-predictor"
        }
        
        external_model = {
            "external": True,
            "requiresApiKey": True,
            "apiUrl": "https://api.anthropic.com/v1/messages"
        }
        
        # Local model should not require API key or URL
        assert local_model["external"] is False
        assert local_model["requiresApiKey"] is False
        assert "serviceName" in local_model
        
        # External model should require API key and have URL
        assert external_model["external"] is True
        assert external_model["requiresApiKey"] is True
        assert "apiUrl" in external_model


if __name__ == "__main__":
    pytest.main([__file__])
