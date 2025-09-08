"""Pure Prometheus MCP Tools for Chat with Prometheus.

This module provides MCP tools for direct Prometheus interaction:
- search_metrics: Search metrics by name pattern
- get_metric_metadata: Get metric help text and type
- get_label_values: Get available label values for a metric
- execute_promql: Execute PromQL query
- explain_results: Explain query results in natural language
- suggest_queries: Suggest related queries based on user intent

These tools enable LLMs to interact directly with Prometheus
without requiring a pre-built knowledge base.
"""

import json
import logging
from typing import Dict, Any, List, Optional, Union
import requests
from datetime import datetime, timedelta
import re

from core.config import PROMETHEUS_URL, THANOS_TOKEN, VERIFY_SSL
from core.llm_client import summarize_with_llm


def _resp(content: str, is_error: bool = False) -> List[Dict[str, Any]]:
    """Helper to format MCP tool responses consistently."""
    return [{"type": "text", "text": content}]


def _make_prometheus_request(endpoint: str, params: Optional[Dict] = None) -> Dict[str, Any]:
    """Make authenticated request to Prometheus/Thanos."""
    url = f"{PROMETHEUS_URL}{endpoint}"
    headers = {}
    
    if THANOS_TOKEN:
        headers["Authorization"] = f"Bearer {THANOS_TOKEN}"
    
    try:
        response = requests.get(
            url, 
            params=params, 
            headers=headers, 
            verify=VERIFY_SSL,
            timeout=30
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        logging.error(f"Prometheus request failed: {e}")
        raise


def search_metrics(
    pattern: str = "",
    limit: int = 50
) -> List[Dict[str, Any]]:
    """Search for metrics by name pattern.
    
    Args:
        pattern: Regex pattern to match metric names (e.g., "cpu", "memory", "vllm")
        limit: Maximum number of metrics to return
    
    Returns:
        List of matching metrics with their names and basic info
    """
    try:
        # Get all metric names
        response = _make_prometheus_request("/api/v1/label/__name__/values")
        all_metrics = response.get("data", [])
        
        # Filter by pattern if provided
        if pattern:
            regex = re.compile(pattern, re.IGNORECASE)
            matching_metrics = [m for m in all_metrics if regex.search(m)]
        else:
            matching_metrics = all_metrics
        
        # Limit results
        matching_metrics = matching_metrics[:limit]
        
        # Get basic info for each metric
        metrics_info = []
        for metric in matching_metrics:
            try:
                metadata_response = _make_prometheus_request(
                    "/api/v1/metadata", 
                    {"metric": metric}
                )
                metadata = metadata_response.get("data", {}).get(metric, [])
                
                if metadata:
                    metric_info = {
                        "name": metric,
                        "type": metadata[0].get("type", "unknown"),
                        "help": metadata[0].get("help", "No description available"),
                        "unit": metadata[0].get("unit", "")
                    }
                else:
                    metric_info = {
                        "name": metric,
                        "type": "unknown",
                        "help": "No description available",
                        "unit": ""
                    }
                
                metrics_info.append(metric_info)
            except Exception as e:
                logging.warning(f"Failed to get metadata for {metric}: {e}")
                metrics_info.append({
                    "name": metric,
                    "type": "unknown", 
                    "help": "No description available",
                    "unit": ""
                })
        
        result = {
            "total_found": len(matching_metrics),
            "metrics": metrics_info,
            "pattern": pattern,
            "limit": limit
        }
        
        return _resp(json.dumps(result, indent=2))
        
    except Exception as e:
        logging.error(f"Error searching metrics: {e}")
        return _resp(f"Error searching metrics: {str(e)}", is_error=True)


def get_metric_metadata(
    metric_name: str
) -> List[Dict[str, Any]]:
    """Get detailed metadata for a specific metric.
    
    Args:
        metric_name: Name of the metric to get metadata for
    
    Returns:
        Detailed metadata including type, help text, unit, and available labels
    """
    try:
        # Get metric metadata
        metadata_response = _make_prometheus_request(
            "/api/v1/metadata", 
            {"metric": metric_name}
        )
        metadata = metadata_response.get("data", {}).get(metric_name, [])
        
        if not metadata:
            return _resp(f"Metric '{metric_name}' not found", is_error=True)
        
        # Get available labels for this metric
        labels_response = _make_prometheus_request("/api/v1/labels")
        all_labels = labels_response.get("data", [])
        
        # Get sample values for common labels
        sample_values = {}
        for label in ["instance", "job", "namespace", "pod", "node"]:
            if label in all_labels:
                try:
                    values_response = _make_prometheus_request(
                        f"/api/v1/label/{label}/values"
                    )
                    sample_values[label] = values_response.get("data", [])[:10]  # Limit to 10 samples
                except Exception:
                    sample_values[label] = []
        
        result = {
            "metric_name": metric_name,
            "metadata": metadata[0] if metadata else {},
            "available_labels": all_labels,
            "sample_label_values": sample_values,
            "query_examples": _generate_query_examples(metric_name, metadata[0] if metadata else {})
        }
        
        return _resp(json.dumps(result, indent=2))
        
    except Exception as e:
        logging.error(f"Error getting metric metadata: {e}")
        return _resp(f"Error getting metric metadata: {str(e)}", is_error=True)


def get_label_values(
    metric_name: str,
    label_name: str
) -> List[Dict[str, Any]]:
    """Get all possible values for a specific label of a metric.
    
    Args:
        metric_name: Name of the metric
        label_name: Name of the label to get values for
    
    Returns:
        List of all possible values for the label
    """
    try:
        response = _make_prometheus_request(f"/api/v1/label/{label_name}/values")
        values = response.get("data", [])
        
        result = {
            "metric_name": metric_name,
            "label_name": label_name,
            "values": values,
            "count": len(values)
        }
        
        return _resp(json.dumps(result, indent=2))
        
    except Exception as e:
        logging.error(f"Error getting label values: {e}")
        return _resp(f"Error getting label values: {str(e)}", is_error=True)


def execute_promql(
    query: str,
    time_range: Optional[str] = None,
    start_datetime: Optional[str] = None,
    end_datetime: Optional[str] = None
) -> List[Dict[str, Any]]:
    """Execute a PromQL query and return results.
    
    Args:
        query: PromQL query to execute
        time_range: Natural language time range (e.g., "last 5 minutes", "1 hour ago")
        start_datetime: Start time in ISO format
        end_datetime: End time in ISO format
    
    Returns:
        Query results with metadata and explanation
    """
    try:
        # Determine query type and time parameters
        if time_range:
            # For range queries, use last 5 minutes as default
            end_time = datetime.now()
            start_time = end_time - timedelta(minutes=5)
        elif start_datetime and end_datetime:
            start_time = datetime.fromisoformat(start_datetime.replace('Z', '+00:00'))
            end_time = datetime.fromisoformat(end_datetime.replace('Z', '+00:00'))
        else:
            # Default to instant query (current time)
            end_time = datetime.now()
            start_time = None
        
        # Execute the query
        if start_time:
            # Range query
            params = {
                "query": query,
                "start": start_time.timestamp(),
                "end": end_time.timestamp(),
                "step": "15s"  # 15-second resolution
            }
            endpoint = "/api/v1/query_range"
        else:
            # Instant query
            params = {"query": query}
            endpoint = "/api/v1/query"
        
        response = _make_prometheus_request(endpoint, params)
        
        # Parse results
        result_type = response.get("data", {}).get("resultType", "")
        results = response.get("data", {}).get("result", [])
        
        # Format results for better readability
        formatted_results = []
        for result in results:
            if result_type == "vector":
                # Instant query result
                formatted_result = {
                    "metric": result.get("metric", {}),
                    "value": result.get("value", [])
                }
            elif result_type == "matrix":
                # Range query result
                formatted_result = {
                    "metric": result.get("metric", {}),
                    "values": result.get("values", [])
                }
            else:
                formatted_result = result
            
            formatted_results.append(formatted_result)
        
        # Generate explanation
        explanation = _explain_query_results(query, formatted_results, result_type)
        
        result = {
            "query": query,
            "result_type": result_type,
            "results": formatted_results,
            "result_count": len(formatted_results),
            "explanation": explanation,
            "time_range": {
                "start": start_time.isoformat() if start_time else None,
                "end": end_time.isoformat()
            }
        }
        
        return _resp(json.dumps(result, indent=2))
        
    except Exception as e:
        logging.error(f"Error executing PromQL: {e}")
        return _resp(f"Error executing PromQL: {str(e)}", is_error=True)


def explain_results(
    query: str,
    results: List[Dict[str, Any]],
    result_type: str = "vector"
) -> List[Dict[str, Any]]:
    """Explain PromQL query results in natural language.
    
    Args:
        query: The PromQL query that was executed
        results: The query results
        result_type: Type of results (vector, matrix, scalar, string)
    
    Returns:
        Natural language explanation of the results
    """
    try:
        explanation = _explain_query_results(query, results, result_type)
        
        result = {
            "query": query,
            "result_type": result_type,
            "explanation": explanation,
            "result_count": len(results)
        }
        
        return _resp(json.dumps(result, indent=2))
        
    except Exception as e:
        logging.error(f"Error explaining results: {e}")
        return _resp(f"Error explaining results: {str(e)}", is_error=True)


def suggest_queries(
    user_intent: str,
    context: Optional[str] = None
) -> List[Dict[str, Any]]:
    """Suggest related PromQL queries based on user intent.
    
    Args:
        user_intent: What the user wants to know (e.g., "CPU usage", "memory problems")
        context: Additional context (e.g., "OpenShift cluster", "ML workloads")
    
    Returns:
        List of suggested PromQL queries with explanations
    """
    try:
        # This would ideally use an LLM to generate suggestions
        # For now, we'll provide some basic suggestions based on common patterns
        
        suggestions = _generate_query_suggestions(user_intent, context)
        
        result = {
            "user_intent": user_intent,
            "context": context,
            "suggestions": suggestions
        }
        
        return _resp(json.dumps(result, indent=2))
        
    except Exception as e:
        logging.error(f"Error suggesting queries: {e}")
        return _resp(f"Error suggesting queries: {str(e)}", is_error=True)


# Helper functions

def _generate_query_examples(metric_name: str, metadata: Dict[str, Any]) -> List[str]:
    """Generate example PromQL queries for a metric."""
    examples = []
    metric_type = metadata.get("type", "unknown")
    
    if metric_type == "counter":
        examples.extend([
            f"rate({metric_name}[5m])",
            f"increase({metric_name}[1h])",
            f"sum(rate({metric_name}[5m]))"
        ])
    elif metric_type == "gauge":
        examples.extend([
            f"avg({metric_name})",
            f"max({metric_name})",
            f"min({metric_name})"
        ])
    elif metric_type == "histogram":
        examples.extend([
            f"histogram_quantile(0.95, rate({metric_name}_bucket[5m]))",
            f"histogram_quantile(0.50, rate({metric_name}_bucket[5m]))",
            f"rate({metric_name}_sum[5m])"
        ])
    else:
        examples.append(metric_name)
    
    return examples


def _explain_query_results(query: str, results: List[Dict[str, Any]], result_type: str) -> str:
    """Generate natural language explanation of query results."""
    if not results:
        return "No results found for this query."
    
    if result_type == "vector":
        if len(results) == 1:
            result = results[0]
            metric = result.get("metric", {})
            value = result.get("value", [])
            if value and len(value) >= 2:
                return f"Found 1 result: {metric.get('__name__', 'metric')} = {value[1]} at {datetime.fromtimestamp(float(value[0]))}"
        else:
            return f"Found {len(results)} results for this query."
    
    elif result_type == "matrix":
        return f"Found {len(results)} time series with data points over the specified time range."
    
    else:
        return f"Query returned {len(results)} results of type {result_type}."


def _generate_query_suggestions(user_intent: str, context: Optional[str]) -> List[Dict[str, Any]]:
    """Generate query suggestions based on user intent."""
    suggestions = []
    
    intent_lower = user_intent.lower()
    
    if "cpu" in intent_lower:
        suggestions.extend([
            {
                "query": "100 - (avg(rate(node_cpu_seconds_total{mode=\"idle\"}[5m])) * 100)",
                "description": "CPU usage percentage",
                "explanation": "Shows CPU utilization as a percentage"
            },
            {
                "query": "rate(node_cpu_seconds_total[5m])",
                "description": "CPU time per second",
                "explanation": "Shows CPU time consumption rate"
            }
        ])
    
    if "memory" in intent_lower:
        suggestions.extend([
            {
                "query": "(node_memory_MemTotal_bytes - node_memory_MemAvailable_bytes) / node_memory_MemTotal_bytes * 100",
                "description": "Memory usage percentage",
                "explanation": "Shows memory utilization as a percentage"
            },
            {
                "query": "node_memory_MemAvailable_bytes",
                "description": "Available memory",
                "explanation": "Shows available memory in bytes"
            }
        ])
    
    if "pod" in intent_lower or "container" in intent_lower:
        suggestions.extend([
            {
                "query": "kube_pod_status_phase",
                "description": "Pod status",
                "explanation": "Shows current phase of pods (Running, Pending, Failed, etc.)"
            },
            {
                "query": "kube_deployment_status_replicas_ready",
                "description": "Ready replicas",
                "explanation": "Shows number of ready replicas for deployments"
            }
        ])
    
    if "gpu" in intent_lower or "vllm" in intent_lower:
        suggestions.extend([
            {
                "query": "avg(DCGM_FI_DEV_GPU_UTIL)",
                "description": "GPU utilization",
                "explanation": "Shows average GPU utilization percentage"
            },
            {
                "query": "histogram_quantile(0.95, rate(vllm:e2e_request_latency_seconds_bucket[5m]))",
                "description": "ML inference latency",
                "explanation": "Shows 95th percentile latency for ML inference"
            }
        ])
    
    # If no specific suggestions, provide general ones
    if not suggestions:
        suggestions.extend([
            {
                "query": "up",
                "description": "Service availability",
                "explanation": "Shows if services are running (1 = up, 0 = down)"
            },
            {
                "query": "ALERTS",
                "description": "Active alerts",
                "explanation": "Shows currently active alerts"
            }
        ])
    
    return suggestions


def select_best_metric(
    user_intent: str,
    available_metrics: List[str],
    context: Optional[str] = None
) -> List[Dict[str, Any]]:
    """LLM-powered metric selection based on user intent.
    
    This tool uses LLM intelligence to select the most relevant metric
    from a list of available metrics based on the user's intent.
    
    Args:
        user_intent: What the user wants to know (e.g., "GPU utilization", "memory usage")
        available_metrics: List of metric names to choose from
        context: Additional context (e.g., "OpenShift cluster", "ML workloads")
    
    Returns:
        The most relevant metric name with explanation
    """
    try:
        if not available_metrics:
            return _resp("No metrics available to select from", is_error=True)
        
        if len(available_metrics) == 1:
            return _resp(json.dumps({
                "selected_metric": available_metrics[0],
                "reasoning": "Only one metric available",
                "confidence": 1.0
            }, indent=2))
        
        # Create prompt for LLM
        prompt = f"""
        You are an expert in Prometheus metrics and observability. 
        
        User wants to know: "{user_intent}"
        Context: {context or "general observability"}
        Available metrics: {', '.join(available_metrics)}
        
        Select the most relevant metric for the user's intent. Consider:
        
        1. **Semantic meaning**: Match the user's intent with metric names
        2. **Domain knowledge**: 
           - DCGM_* = GPU metrics (DCGM_FI_DEV_GPU_UTIL = GPU utilization)
           - vLLM:* = ML inference metrics (vLLM:e2e_request_latency_seconds = latency)
           - node_* = system metrics (node_cpu_seconds_total = CPU, node_memory_* = memory)
           - kube_* = Kubernetes metrics (kube_pod_status_phase = pod status)
           - ALERTS = active alerts
        3. **Keyword matching**: Look for keywords in metric names that match user intent
        4. **Metric type relevance**: Prefer metrics that directly answer the user's question
        
        Examples:
        - "GPU utilization" → DCGM_FI_DEV_GPU_UTIL
        - "memory usage" → node_memory_MemTotal_bytes or node_memory_MemAvailable_bytes
        - "CPU usage" → node_cpu_seconds_total
        - "pod status" → kube_pod_status_phase
        - "active alerts" → ALERTS
        
        Return your response in this exact JSON format:
        {{
            "selected_metric": "metric_name",
            "reasoning": "Brief explanation of why this metric was selected",
            "confidence": 0.95
        }}
        
        Only return the JSON, nothing else.
        """
        
        # Use LLM to select best metric
        try:
            llm_response = summarize_with_llm(prompt)
            
            # Parse LLM response
            if llm_response and isinstance(llm_response, str):
                # Try to extract JSON from response
                import re
                json_match = re.search(r'\{.*\}', llm_response, re.DOTALL)
                if json_match:
                    json_str = json_match.group(0)
                    selection_result = json.loads(json_str)
                    
                    # Validate selection
                    selected_metric = selection_result.get("selected_metric", "")
                    if selected_metric in available_metrics:
                        return _resp(json.dumps(selection_result, indent=2))
                    else:
                        # Fallback to first metric if selection is invalid
                        return _resp(json.dumps({
                            "selected_metric": available_metrics[0],
                            "reasoning": f"LLM selected '{selected_metric}' which is not in available metrics. Using first available metric.",
                            "confidence": 0.5
                        }, indent=2))
                else:
                    # If no JSON found, try to extract metric name from text
                    for metric in available_metrics:
                        if metric.lower() in llm_response.lower():
                            return _resp(json.dumps({
                                "selected_metric": metric,
                                "reasoning": f"Found '{metric}' mentioned in LLM response",
                                "confidence": 0.7
                            }, indent=2))
            
        except Exception as llm_error:
            logging.warning(f"LLM selection failed: {llm_error}")
        
        # Fallback: Use simple keyword matching
        selected_metric = _fallback_metric_selection(user_intent, available_metrics)
        
        return _resp(json.dumps({
            "selected_metric": selected_metric,
            "reasoning": "Used fallback keyword matching due to LLM unavailability",
            "confidence": 0.6
        }, indent=2))
        
    except Exception as e:
        logging.error(f"Error selecting metric: {e}")
        return _resp(f"Error selecting metric: {str(e)}", is_error=True)


def _fallback_metric_selection(user_intent: str, available_metrics: List[str]) -> str:
    """Fallback metric selection using simple keyword matching.
    
    This is used when LLM selection fails or is unavailable.
    """
    intent_lower = user_intent.lower()
    
    # Define keyword mappings
    keyword_mappings = {
        # GPU metrics
        "gpu": ["DCGM_FI_DEV_GPU_UTIL", "DCGM_FI_DEV_GPU_TEMP", "DCGM_FI_DEV_GPU_MEMORY"],
        "utilization": ["DCGM_FI_DEV_GPU_UTIL", "node_cpu_seconds_total"],
        "temperature": ["DCGM_FI_DEV_GPU_TEMP", "node_hwmon_temp_celsius"],
        
        # Memory metrics
        "memory": ["node_memory_MemTotal_bytes", "node_memory_MemAvailable_bytes", "node_memory_MemFree_bytes"],
        "ram": ["node_memory_MemTotal_bytes", "node_memory_MemAvailable_bytes"],
        
        # CPU metrics
        "cpu": ["node_cpu_seconds_total", "node_load1", "node_load5", "node_load15"],
        
        # Kubernetes metrics
        "pod": ["kube_pod_status_phase", "kube_pod_info"],
        "container": ["kube_pod_container_status_ready", "kube_pod_container_status_running"],
        "deployment": ["kube_deployment_status_replicas_ready", "kube_deployment_status_replicas"],
        
        # Alert metrics
        "alert": ["ALERTS", "ALERTS_FOR_STATE"],
        "alerts": ["ALERTS", "ALERTS_FOR_STATE"],
        
        # vLLM metrics
        "vllm": ["vllm:e2e_request_latency_seconds", "vllm:time_to_first_token_seconds"],
        "latency": ["vllm:e2e_request_latency_seconds", "vllm:time_to_first_token_seconds"],
        "inference": ["vllm:e2e_request_latency_seconds", "vllm:time_to_first_token_seconds"],
        
        # System metrics
        "disk": ["node_filesystem_size_bytes", "node_filesystem_free_bytes"],
        "network": ["node_network_receive_bytes_total", "node_network_transmit_bytes_total"],
        "load": ["node_load1", "node_load5", "node_load15"]
    }
    
    # Find best matching metric
    best_match = None
    best_score = 0
    
    for keyword, preferred_metrics in keyword_mappings.items():
        if keyword in intent_lower:
            # Check if any preferred metrics are available
            for metric in preferred_metrics:
                if metric in available_metrics:
                    # Calculate score based on keyword relevance
                    score = len(keyword) / len(intent_lower)  # Simple scoring
                    if score > best_score:
                        best_score = score
                        best_match = metric
    
    # If no keyword match found, try partial matching
    if not best_match:
        for metric in available_metrics:
            metric_lower = metric.lower()
            # Check if any word from user intent appears in metric name
            intent_words = intent_lower.split()
            for word in intent_words:
                if len(word) > 2 and word in metric_lower:  # Avoid very short words
                    return metric
    
    # Return best match or first available metric
    return best_match if best_match else available_metrics[0]
