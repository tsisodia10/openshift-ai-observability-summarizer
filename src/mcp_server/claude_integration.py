#!/usr/bin/env python3
"""
Claude Integration for Prometheus Chat Bot

This module integrates Claude AI with MCP tools to provide intelligent
Prometheus querying and analysis through natural language.
"""

import anthropic
import os
import json
import logging
from typing import Optional, Dict, Any
import sys
import os
import importlib.util

# Add current directory to path to ensure local imports work
current_dir = os.path.dirname(__file__)
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Import the local observability_mcp module directly to avoid conflicts with the installed mcp package
mcp_path = os.path.join(os.path.dirname(__file__), 'observability_mcp.py')
spec = importlib.util.spec_from_file_location("local_observability_mcp", mcp_path)
local_mcp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(local_mcp)
ObservabilityMCPServer = local_mcp.ObservabilityMCPServer

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PrometheusChatBot:
    """AI-powered Prometheus chat bot using Claude and MCP tools."""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize the chat bot.
        
        Args:
            api_key: Anthropic API key. If None, will try to get from environment.
        """
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("Anthropic API key is required. Set ANTHROPIC_API_KEY environment variable.")
        
        # Initialize Claude client
        self.claude_client = anthropic.Anthropic(api_key=self.api_key)
        
        # Initialize MCP server (our Prometheus tools)
        self.mcp_server = ObservabilityMCPServer()
        
        logger.info("PrometheusChatBot initialized successfully")
    
    def _create_system_prompt(self, namespace: Optional[str] = None, scope: str = "cluster") -> str:
        """Create system prompt for Claude based on context."""
        return f"""
You are an expert Prometheus observability assistant. You have access to these powerful tools:

**Available Tools:**
- search_metrics(pattern): Search for metrics by pattern (e.g., "memory", "cpu", "gpu")
- execute_promql(query): Execute PromQL queries and get results
- get_metric_metadata(metric): Get detailed information about a specific metric
- explain_results(data): Explain query results in human-readable format
- suggest_queries(intent): Suggest related queries based on user intent
- select_best_metric(intent, metrics): Intelligently select the most relevant metric

**Current Context:**
- Scope: {scope}
- Namespace: {namespace if namespace else 'cluster-wide'}
- Environment: OpenShift cluster with AI/ML workloads

**Your Role:**
1. Understand user questions about Prometheus metrics
2. Use the appropriate tools to gather data
3. Provide intelligent analysis and insights
4. Suggest follow-up questions or actions
5. Always be helpful and accurate

**Response Style:**
- Be conversational and friendly
- Provide clear explanations
- Include relevant metrics and data
- Suggest next steps when appropriate
- Use emojis sparingly for better readability

**Important:**
- Always use the tools to get real data
- Don't make up or guess metric values
- If you can't find relevant metrics, say so clearly
- Provide actionable insights based on the data
"""
    
    def chat(self, user_question: str, namespace: Optional[str] = None, scope: str = "cluster") -> str:
        """Chat with the user about Prometheus metrics.
        
        Args:
            user_question: The user's question about metrics
            namespace: Target namespace (if scope is "namespace")
            scope: Either "cluster" or "namespace"
            
        Returns:
            Claude's response with analysis and insights
        """
        try:
            # Create system prompt
            system_prompt = self._create_system_prompt(namespace, scope)
            
            # Add context to user question
            context_question = f"""
User Question: {user_question}

Context:
- Scope: {scope}
- Namespace: {namespace if namespace else 'cluster-wide'}
- User wants to understand: {user_question}

Please help analyze this request using the available Prometheus tools.
"""
            
            # Get response from Claude
            response = self.claude_client.messages.create(
                model="claude-3-sonnet-20240229",
                system=system_prompt,
                messages=[{"role": "user", "content": context_question}],
                max_tokens=1500
            )
            
            return response.content[0].text
            
        except Exception as e:
            logger.error(f"Error in chat: {e}")
            return f"Sorry, I encountered an error: {str(e)}. Please try again."
    
    def get_available_tools(self) -> list:
        """Get list of available MCP tools."""
        return [
            "search_metrics",
            "execute_promql", 
            "get_metric_metadata",
            "explain_results",
            "suggest_queries",
            "select_best_metric"
        ]
    
    def test_connection(self) -> bool:
        """Test if the integration is working properly."""
        try:
            # Test Claude API
            test_response = self.claude_client.messages.create(
                model="claude-3-sonnet-20240229",
                messages=[{"role": "user", "content": "Hello, are you working?"}],
                max_tokens=50
            )
            
            # Test MCP server
            if hasattr(self.mcp_server, 'mcp'):
                logger.info("Claude API and MCP server are working")
                return True
            else:
                logger.error("MCP server not properly initialized")
                return False
                
        except Exception as e:
            logger.error(f"Connection test failed: {e}")
            return False

# Example usage
if __name__ == "__main__":
    # Test the integration
    try:
        bot = PrometheusChatBot()
        if bot.test_connection():
            print("✅ Integration test passed!")
            response = bot.chat("What metrics are available?", scope="cluster")
            print(f"Response: {response}")
        else:
            print("❌ Integration test failed!")
    except Exception as e:
        print(f"❌ Error: {e}")
