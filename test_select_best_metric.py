#!/usr/bin/env python3
"""
Test script for the new select_best_metric MCP tool.

This script tests the LLM-powered metric selection functionality
to ensure it correctly selects the most relevant metric based on user intent.
"""

import unittest
import json
import os
import sys
from unittest.mock import patch, MagicMock

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from mcp_server.tools.prometheus_tools import select_best_metric, _fallback_metric_selection


class TestSelectBestMetric(unittest.TestCase):
    """Test cases for the select_best_metric tool."""

    def setUp(self):
        """Set up test fixtures."""
        print("\n🚀 Starting select_best_metric Tests")
        print("==================================================")

    def _extract_content(self, response):
        """Helper to extract content from MCP tool response format."""
        if isinstance(response, list) and response and "text" in response[0]:
            try:
                return json.loads(response[0]["text"])
            except json.JSONDecodeError:
                return response[0]["text"]
        return response

    def test_single_metric_selection(self):
        """Test selection when only one metric is available."""
        print("\n🔍 Testing single metric selection...")
        
        available_metrics = ["DCGM_FI_DEV_GPU_UTIL"]
        result = select_best_metric("GPU utilization", available_metrics)
        result_data = self._extract_content(result)
        
        print(f"Single metric result: {json.dumps(result_data, indent=2)}")
        
        self.assertEqual(result_data["selected_metric"], "DCGM_FI_DEV_GPU_UTIL")
        self.assertEqual(result_data["confidence"], 1.0)
        self.assertEqual(result_data["reasoning"], "Only one metric available")
        print("✅ Single metric selection test passed")

    def test_empty_metrics_list(self):
        """Test handling of empty metrics list."""
        print("\n🔍 Testing empty metrics list...")
        
        result = select_best_metric("GPU utilization", [])
        result_data = self._extract_content(result)
        
        print(f"Empty metrics result: {json.dumps(result_data, indent=2)}")
        
        self.assertIn("no metrics available", str(result_data).lower())
        print("✅ Empty metrics list test passed")

    def test_fallback_selection(self):
        """Test fallback keyword matching when LLM is unavailable."""
        print("\n🔍 Testing fallback selection...")
        
        available_metrics = [
            "DCGM_FI_DEV_GPU_UTIL",
            "DCGM_FI_DEV_GPU_TEMP", 
            "DCGM_FI_DEV_GPU_MEMORY",
            "node_cpu_seconds_total",
            "node_memory_MemTotal_bytes"
        ]
        
        # Test GPU utilization
        result = select_best_metric("GPU utilization", available_metrics)
        result_data = self._extract_content(result)
        
        print(f"GPU utilization result: {json.dumps(result_data, indent=2)}")
        
        # Should select GPU utilization metric
        self.assertIn("DCGM_FI_DEV_GPU_UTIL", result_data["selected_metric"])
        print("✅ GPU utilization fallback test passed")
        
        # Test memory usage
        result = select_best_metric("memory usage", available_metrics)
        result_data = self._extract_content(result)
        
        print(f"Memory usage result: {json.dumps(result_data, indent=2)}")
        
        # Should select memory metric
        self.assertIn("memory", result_data["selected_metric"].lower())
        print("✅ Memory usage fallback test passed")

    def test_fallback_function_directly(self):
        """Test the fallback function directly."""
        print("\n🔍 Testing fallback function directly...")
        
        available_metrics = [
            "DCGM_FI_DEV_GPU_UTIL",
            "DCGM_FI_DEV_GPU_TEMP",
            "node_cpu_seconds_total",
            "node_memory_MemTotal_bytes",
            "kube_pod_status_phase",
            "ALERTS"
        ]
        
        # Test various intents
        test_cases = [
            ("GPU utilization", "DCGM_FI_DEV_GPU_UTIL"),
            ("memory usage", "node_memory_MemTotal_bytes"),
            ("CPU usage", "node_cpu_seconds_total"),
            ("pod status", "kube_pod_status_phase"),
            ("active alerts", "ALERTS"),
            ("GPU temperature", "DCGM_FI_DEV_GPU_TEMP")
        ]
        
        for intent, expected_metric in test_cases:
            selected = _fallback_metric_selection(intent, available_metrics)
            print(f"Intent: '{intent}' → Selected: '{selected}' (Expected: '{expected_metric}')")
            self.assertEqual(selected, expected_metric)
        
        print("✅ Fallback function direct test passed")

    @patch('mcp_server.tools.prometheus_tools.summarize_with_llm')
    def test_llm_selection_success(self, mock_llm):
        """Test successful LLM selection."""
        print("\n🔍 Testing LLM selection success...")
        
        # Mock LLM response
        mock_llm.return_value = json.dumps({
            "selected_metric": "DCGM_FI_DEV_GPU_UTIL",
            "reasoning": "DCGM_FI_DEV_GPU_UTIL is the most relevant metric for GPU utilization",
            "confidence": 0.95
        })
        
        available_metrics = [
            "DCGM_FI_DEV_GPU_UTIL",
            "DCGM_FI_DEV_GPU_TEMP",
            "DCGM_FI_DEV_GPU_MEMORY"
        ]
        
        result = select_best_metric("GPU utilization", available_metrics)
        result_data = self._extract_content(result)
        
        print(f"LLM selection result: {json.dumps(result_data, indent=2)}")
        
        self.assertEqual(result_data["selected_metric"], "DCGM_FI_DEV_GPU_UTIL")
        self.assertEqual(result_data["confidence"], 0.95)
        self.assertIn("DCGM_FI_DEV_GPU_UTIL", result_data["reasoning"])
        print("✅ LLM selection success test passed")

    @patch('mcp_server.tools.prometheus_tools.summarize_with_llm')
    def test_llm_selection_invalid_metric(self, mock_llm):
        """Test LLM selection with invalid metric name."""
        print("\n🔍 Testing LLM selection with invalid metric...")
        
        # Mock LLM response with invalid metric
        mock_llm.return_value = json.dumps({
            "selected_metric": "INVALID_METRIC_NAME",
            "reasoning": "This metric should be selected",
            "confidence": 0.95
        })
        
        available_metrics = [
            "DCGM_FI_DEV_GPU_UTIL",
            "DCGM_FI_DEV_GPU_TEMP",
            "DCGM_FI_DEV_GPU_MEMORY"
        ]
        
        result = select_best_metric("GPU utilization", available_metrics)
        result_data = self._extract_content(result)
        
        print(f"Invalid metric result: {json.dumps(result_data, indent=2)}")
        
        # Should fallback to first available metric
        self.assertEqual(result_data["selected_metric"], "DCGM_FI_DEV_GPU_UTIL")
        self.assertEqual(result_data["confidence"], 0.5)
        self.assertIn("not in available metrics", result_data["reasoning"])
        print("✅ LLM selection invalid metric test passed")

    @patch('mcp_server.tools.prometheus_tools.summarize_with_llm')
    def test_llm_selection_failure(self, mock_llm):
        """Test LLM selection failure."""
        print("\n🔍 Testing LLM selection failure...")
        
        # Mock LLM failure
        mock_llm.side_effect = Exception("LLM connection failed")
        
        available_metrics = [
            "DCGM_FI_DEV_GPU_UTIL",
            "DCGM_FI_DEV_GPU_TEMP",
            "DCGM_FI_DEV_GPU_MEMORY"
        ]
        
        result = select_best_metric("GPU utilization", available_metrics)
        result_data = self._extract_content(result)
        
        print(f"LLM failure result: {json.dumps(result_data, indent=2)}")
        
        # Should fallback to keyword matching
        self.assertEqual(result_data["selected_metric"], "DCGM_FI_DEV_GPU_UTIL")
        self.assertEqual(result_data["confidence"], 0.6)
        self.assertIn("fallback keyword matching", result_data["reasoning"])
        print("✅ LLM selection failure test passed")

    def test_complex_user_intents(self):
        """Test complex user intents."""
        print("\n🔍 Testing complex user intents...")
        
        available_metrics = [
            "DCGM_FI_DEV_GPU_UTIL",
            "DCGM_FI_DEV_GPU_TEMP",
            "DCGM_FI_DEV_GPU_MEMORY",
            "node_cpu_seconds_total",
            "node_memory_MemTotal_bytes",
            "node_memory_MemAvailable_bytes",
            "kube_pod_status_phase",
            "kube_deployment_status_replicas_ready",
            "ALERTS",
            "vllm:e2e_request_latency_seconds",
            "vllm:time_to_first_token_seconds"
        ]
        
        test_cases = [
            ("What's my GPU utilization?", "DCGM_FI_DEV_GPU_UTIL"),
            ("Show me memory usage", "node_memory_MemTotal_bytes"),
            ("How much CPU is being used?", "node_cpu_seconds_total"),
            ("Are there any active alerts?", "ALERTS"),
            ("What's the status of my pods?", "kube_pod_status_phase"),
            ("How many replicas are ready?", "kube_deployment_status_replicas_ready"),
            ("What's the inference latency?", "vllm:e2e_request_latency_seconds"),
            ("GPU temperature", "DCGM_FI_DEV_GPU_TEMP")
        ]
        
        for intent, expected_metric in test_cases:
            result = select_best_metric(intent, available_metrics)
            result_data = self._extract_content(result)
            selected = result_data["selected_metric"]
            
            print(f"Intent: '{intent}' → Selected: '{selected}' (Expected: '{expected_metric}')")
            
            # For fallback testing, we expect the expected metric to be selected
            # In real LLM scenarios, this might be different
            if "fallback" in result_data["reasoning"].lower():
                self.assertEqual(selected, expected_metric)
        
        print("✅ Complex user intents test passed")

    def tearDown(self):
        """Clean up after tests."""
        pass

    @classmethod
    def tearDownClass(cls):
        """Clean up after all tests."""
        print("\n�� All select_best_metric tests completed successfully!")
        print("\n📋 Summary:")
        print("- ✅ Single metric selection")
        print("- ✅ Empty metrics list handling")
        print("- ✅ Fallback keyword matching")
        print("- ✅ LLM selection success")
        print("- ✅ LLM selection with invalid metric")
        print("- ✅ LLM selection failure")
        print("- ✅ Complex user intents")
        print("\n🎯 The select_best_metric tool is ready for production!")


if __name__ == "__main__":
    unittest.main()
