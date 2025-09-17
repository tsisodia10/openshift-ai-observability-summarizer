import logging

from .settings import settings
from common.pylogger import get_python_logger, force_reconfigure_all_loggers

try:
    from .settings import settings
except ImportError:
    from settings import settings
try:
    from mcp_server.utils.pylogger import get_python_logger, force_reconfigure_all_loggers
except ImportError:
    from utils.pylogger import get_python_logger, force_reconfigure_all_loggers


class ObservabilityMCPServer:
    def __init__(self) -> None:
        # Lazy import to avoid import-time circulars when fastmcp pulls in mcp.types
        from fastmcp import FastMCP  # type: ignore

        get_python_logger(settings.PYTHON_LOG_LEVEL)
        self.mcp = FastMCP("metrics-observability")
        # Ensure third-party loggers are reconfigured after FastMCP init
        force_reconfigure_all_loggers(settings.PYTHON_LOG_LEVEL)
        self._register_mcp_tools()
        logging.getLogger(__name__).info("Observability MCP Server initialized")

    def _register_mcp_tools(self) -> None:
        from .tools.observability_vllm_tools import (
            list_models,
            list_namespaces,
            get_model_config,
            get_vllm_metrics_tool,
            analyze_vllm,
            calculate_metrics,
        )
        from .tools.observability_openshift_tools import (
            analyze_openshift,
            list_openshift_metric_groups,
            list_openshift_namespace_metric_groups,
            chat_openshift,
        )
        from .tools.prometheus_tools import (
            search_metrics,
            get_metric_metadata,
            get_label_values,
            execute_promql,
            explain_results,
            suggest_queries,
            select_best_metric,
        )

        # Existing tools
        self.mcp.tool()(list_models)
        self.mcp.tool()(list_namespaces)
        self.mcp.tool()(get_model_config)
        self.mcp.tool()(get_vllm_metrics_tool)
        self.mcp.tool()(analyze_vllm)
        self.mcp.tool()(calculate_metrics)
        self.mcp.tool()(analyze_openshift)
        self.mcp.tool()(list_openshift_metric_groups)
        self.mcp.tool()(list_openshift_namespace_metric_groups)
        self.mcp.tool()(chat_openshift)
        
        # New Pure Prometheus tools
        self.mcp.tool()(search_metrics)
        self.mcp.tool()(get_metric_metadata)
        self.mcp.tool()(get_label_values)
        self.mcp.tool()(execute_promql)
        self.mcp.tool()(explain_results)
        self.mcp.tool()(suggest_queries)
        self.mcp.tool()(select_best_metric)

