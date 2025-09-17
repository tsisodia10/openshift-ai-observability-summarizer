"""Tests for enhanced Prometheus tools with dynamic metric selection."""

import pytest
import json
from unittest.mock import Mock, patch
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from mcp_server.tools.prometheus_tools import (
    find_best_metric_with_metadata_v2,
    find_best_metric_with_metadata,
    _extract_keywords_for_filtering,
    _score_metric_with_metadata_for_question,
    _rank_metrics_by_relevance,
    _calculate_semantic_score
)


class TestDynamicMetricSelection:
    """Test the dynamic metric selection intelligence."""
    
    def test_extract_keywords_for_filtering(self):
        """Test keyword extraction from user questions."""
        # Test Kubernetes questions
        keywords = _extract_keywords_for_filtering("How many pods are running?")
        assert "pod" in keywords
        
        keywords = _extract_keywords_for_filtering("What services are deployed?")
        assert "service" in keywords
        
        # Test GPU questions
        keywords = _extract_keywords_for_filtering("What's the GPU temperature?")
        assert "gpu" in keywords or "temp" in keywords or "temperature" in keywords
        
        # Test vLLM questions  
        keywords = _extract_keywords_for_filtering("How many tokens generated?")
        assert "vllm" in keywords or "token" in keywords
    
    def test_score_metric_with_metadata_for_question(self):
        """Test metadata-based scoring for metric relevance."""
        # Test pod count question with kube_pod_status_phase
        score = _score_metric_with_metadata_for_question(
            "kube_pod_status_phase",
            "The phase of a pod",
            "gauge",
            "How many pods are running?"
        )
        assert score > 50  # Should get high score for perfect match
        
        # Test GPU temperature question
        score = _score_metric_with_metadata_for_question(
            "DCGM_FI_DEV_GPU_TEMP",
            "GPU temperature in Celsius",
            "gauge", 
            "What's the GPU temperature?"
        )
        assert score > 50  # Should get high score for GPU temp
        
        # Test irrelevant metric
        score = _score_metric_with_metadata_for_question(
            "random_metric",
            "Some random metric",
            "counter",
            "How many pods are running?"
        )
        assert score < 30  # Should get low score
    
    def test_calculate_semantic_score(self):
        """Test semantic scoring for metric names."""
        # Test pod-related scoring
        score = _calculate_semantic_score("pods running", "kube_pod_status_phase")
        assert score > 30
        
        score = _calculate_semantic_score("pods running", "kube_service_info")
        assert score < 20  # Should score lower for non-pod metric
        
        # Test GPU scoring
        score = _calculate_semantic_score("gpu temperature", "DCGM_FI_DEV_GPU_TEMP")
        assert score > 30
    
    def test_rank_metrics_by_relevance(self):
        """Test metric ranking algorithm."""
        test_metrics = [
            "kube_pod_status_phase",
            "kube_service_info", 
            "random_metric",
            "DCGM_FI_DEV_GPU_TEMP"
        ]
        
        ranked = _rank_metrics_by_relevance("pod", test_metrics)
        
        # kube_pod_status_phase should be ranked highly for pod queries
        assert ranked[0] == "kube_pod_status_phase" or ranked[1] == "kube_pod_status_phase"
    
    @patch('mcp_server.tools.prometheus_tools._make_prometheus_request')
    def test_find_best_metric_with_metadata_v2(self, mock_request):
        """Test the v2 metric selection with metadata filtering."""
        # Mock search results
        mock_request.side_effect = [
            # Mock search_metrics response
            {
                "data": {
                    "metrics": [
                        {"name": "kube_pod_status_phase"},
                        {"name": "kube_pod_info"}
                    ]
                }
            },
            # Mock metadata responses
            {
                "data": [{
                    "metric": {"__name__": "kube_pod_status_phase"},
                    "help": "The phase of a pod",
                    "type": "gauge"
                }]
            }
        ]
        
        result = find_best_metric_with_metadata_v2("How many pods are running?")
        
        # Should return structured response with best metric
        result_data = json.loads(result[0]["text"])
        assert "best_metric" in result_data
        assert "suggested_promql" in result_data
    
    @patch('mcp_server.tools.prometheus_tools._make_prometheus_request')
    def test_find_best_metric_with_metadata_gpu_question(self, mock_request):
        """Test metric selection for GPU-related questions."""
        # Mock search results for GPU
        mock_request.side_effect = [
            {
                "data": {
                    "metrics": [
                        {"name": "DCGM_FI_DEV_GPU_TEMP"},
                        {"name": "DCGM_FI_DEV_GPU_UTIL"}
                    ]
                }
            },
            {
                "data": [{
                    "metric": {"__name__": "DCGM_FI_DEV_GPU_TEMP"},
                    "help": "GPU temperature in Celsius",
                    "type": "gauge"
                }]
            }
        ]
        
        result = find_best_metric_with_metadata_v2("What's the GPU temperature?")
        
        result_data = json.loads(result[0]["text"])
        assert "DCGM_FI_DEV_GPU_TEMP" in result_data["best_metric"]
    
    @patch('mcp_server.tools.prometheus_tools._make_prometheus_request')
    def test_find_best_metric_with_metadata_vllm_question(self, mock_request):
        """Test metric selection for vLLM-related questions."""
        # Mock search results for vLLM
        mock_request.side_effect = [
            {
                "data": {
                    "metrics": [
                        {"name": "vllm:generation_tokens_total"},
                        {"name": "vllm:e2e_request_latency_seconds"}
                    ]
                }
            },
            {
                "data": [{
                    "metric": {"__name__": "vllm:generation_tokens_total"},
                    "help": "Total tokens generated by vLLM",
                    "type": "counter"
                }]
            }
        ]
        
        result = find_best_metric_with_metadata_v2("How many tokens generated?")
        
        result_data = json.loads(result[0]["text"])
        assert "vllm:generation_tokens_total" in result_data["best_metric"]


class TestPrometheusToolsIntegration:
    """Integration tests for Prometheus tools."""
    
    @patch('mcp_server.tools.prometheus_tools._make_prometheus_request')
    def test_search_metrics_integration(self, mock_request):
        """Test search_metrics tool integration."""
        from mcp_server.tools.prometheus_tools import search_metrics
        
        mock_request.return_value = {
            "data": {
                "metrics": [
                    {"name": "kube_pod_status_phase"},
                    {"name": "kube_service_info"}
                ]
            }
        }
        
        result = search_metrics("pod")
        
        assert len(result) > 0
        result_data = json.loads(result[0]["text"])
        assert "metrics" in result_data
        assert len(result_data["metrics"]) == 2
    
    @patch('mcp_server.tools.prometheus_tools._make_prometheus_request')
    def test_execute_promql_integration(self, mock_request):
        """Test execute_promql tool integration."""
        from mcp_server.tools.prometheus_tools import execute_promql
        
        mock_request.return_value = {
            "data": {
                "result": [
                    {
                        "metric": {"__name__": "kube_pod_status_phase", "phase": "Running"},
                        "value": [1694123456, "7"]
                    }
                ]
            }
        }
        
        result = execute_promql("sum(kube_pod_status_phase{phase='Running'})")
        
        assert len(result) > 0
        result_data = json.loads(result[0]["text"])
        assert "result" in result_data
    
    @patch('mcp_server.tools.prometheus_tools._make_prometheus_request')
    def test_get_metric_metadata_integration(self, mock_request):
        """Test get_metric_metadata tool integration."""
        from mcp_server.tools.prometheus_tools import get_metric_metadata
        
        mock_request.return_value = {
            "data": [{
                "metric": {"__name__": "kube_pod_status_phase"},
                "help": "The phase of a pod",
                "type": "gauge"
            }]
        }
        
        result = get_metric_metadata("kube_pod_status_phase")
        
        assert len(result) > 0
        result_data = json.loads(result[0]["text"])
        assert "help" in result_data
        assert "type" in result_data


class TestKeywordExtraction:
    """Test keyword extraction for different question types."""
    
    def test_kubernetes_keywords(self):
        """Test extraction of Kubernetes-related keywords."""
        test_cases = [
            ("How many pods are running?", ["pod"]),
            ("What services are deployed?", ["service"]),
            ("Show me node status", ["node"]),
            ("List all deployments", ["deployment"]),
            ("Volume usage statistics", ["volume"])
        ]
        
        for question, expected_keywords in test_cases:
            keywords = _extract_keywords_for_filtering(question)
            for expected in expected_keywords:
                assert expected in keywords, f"Expected '{expected}' in keywords for '{question}'"
    
    def test_gpu_keywords(self):
        """Test extraction of GPU-related keywords."""
        test_cases = [
            ("GPU temperature", ["gpu", "temp", "temperature", "dcgm"]),
            ("Graphics card utilization", ["gpu", "util"]),
            ("NVIDIA power consumption", ["gpu", "power", "dcgm"])
        ]
        
        for question, possible_keywords in test_cases:
            keywords = _extract_keywords_for_filtering(question)
            # At least one of the possible keywords should be present
            assert any(keyword in keywords for keyword in possible_keywords), \
                f"None of {possible_keywords} found in keywords for '{question}'"
    
    def test_vllm_keywords(self):
        """Test extraction of vLLM/AI-related keywords."""
        test_cases = [
            ("Token generation rate", ["vllm", "token"]),
            ("Model inference latency", ["vllm", "model"]),
            ("LLM performance metrics", ["vllm"])
        ]
        
        for question, expected_keywords in test_cases:
            keywords = _extract_keywords_for_filtering(question)
            for expected in expected_keywords:
                assert expected in keywords, f"Expected '{expected}' in keywords for '{question}'"


if __name__ == "__main__":
    pytest.main([__file__])
