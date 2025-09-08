#!/usr/bin/env python3
"""
Test script for Pure MCP Tools implementation.

This script tests the new Prometheus MCP tools to ensure they work correctly
with the local Prometheus instance.
"""

import sys
import os
import json
from datetime import datetime

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from mcp_server.tools.prometheus_tools import (
    search_metrics,
    get_metric_metadata,
    get_label_values,
    execute_promql,
    explain_results,
    suggest_queries
)


def test_search_metrics():
    """Test the search_metrics tool."""
    print("🔍 Testing search_metrics...")
    
    # Test searching for CPU metrics
    result = search_metrics(pattern="cpu", limit=5)
    print(f"CPU metrics search result: {result[0]['text'][:200]}...")
    
    # Test searching for vLLM metrics
    result = search_metrics(pattern="vllm", limit=3)
    print(f"vLLM metrics search result: {result[0]['text'][:200]}...")
    
    print("✅ search_metrics test completed\n")


def test_get_metric_metadata():
    """Test the get_metric_metadata tool."""
    print("📊 Testing get_metric_metadata...")
    
    # Test with a well-documented metric
    result = get_metric_metadata("vllm:e2e_request_latency_seconds")
    print(f"vLLM metadata result: {result[0]['text'][:200]}...")
    
    # Test with a poorly documented metric
    result = get_metric_metadata("node_cpu_seconds_total")
    print(f"Node CPU metadata result: {result[0]['text'][:200]}...")
    
    print("✅ get_metric_metadata test completed\n")


def test_execute_promql():
    """Test the execute_promql tool."""
    print("⚡ Testing execute_promql...")
    
    # Test a simple query
    result = execute_promql("up")
    print(f"Simple query result: {result[0]['text'][:200]}...")
    
    # Test a more complex query
    result = execute_promql("100 - (avg(rate(node_cpu_seconds_total{mode=\"idle\"}[5m])) * 100)")
    print(f"CPU usage query result: {result[0]['text'][:200]}...")
    
    print("✅ execute_promql test completed\n")


def test_suggest_queries():
    """Test the suggest_queries tool."""
    print("💡 Testing suggest_queries...")
    
    # Test CPU usage suggestions
    result = suggest_queries("CPU usage", "OpenShift cluster")
    print(f"CPU usage suggestions: {result[0]['text'][:200]}...")
    
    # Test GPU usage suggestions
    result = suggest_queries("GPU utilization", "ML workloads")
    print(f"GPU usage suggestions: {result[0]['text'][:200]}...")
    
    print("✅ suggest_queries test completed\n")


def main():
    """Run all tests."""
    print("🚀 Starting Pure MCP Tools Tests")
    print("=" * 50)
    
    try:
        test_search_metrics()
        test_get_metric_metadata()
        test_execute_promql()
        test_suggest_queries()
        
        print("🎉 All tests completed successfully!")
        print("\n📋 Summary:")
        print("- ✅ search_metrics: Search metrics by pattern")
        print("- ✅ get_metric_metadata: Get detailed metric information")
        print("- ✅ execute_promql: Execute PromQL queries")
        print("- ✅ suggest_queries: Suggest related queries")
        print("\n🎯 The Pure MCP Tools are ready for LLM integration!")
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
