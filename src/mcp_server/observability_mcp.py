import logging
import os
import sys
import importlib.util

# Add current directory and src directory to path for imports
current_dir = os.path.dirname(__file__)
src_dir = os.path.dirname(current_dir)  # Go up one level to src directory
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

# Import settings using direct file loading to avoid import issues
try:
    from .settings import settings
except ImportError:
    try:
        from settings import settings
    except ImportError:
        # Last resort: load settings directly via importlib
        settings_path = os.path.join(current_dir, 'settings.py')
        if os.path.exists(settings_path):
            spec = importlib.util.spec_from_file_location("settings_local", settings_path)
            settings_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(settings_module)
            settings = settings_module.settings
        else:
            # Fallback to default settings if file not found
            class FallbackSettings:
                PYTHON_LOG_LEVEL = "INFO"
            settings = FallbackSettings()

# Import pylogger with fallback
try:
    from mcp_server.utils.pylogger import get_python_logger, force_reconfigure_all_loggers
except ImportError:
    try:
        from utils.pylogger import get_python_logger, force_reconfigure_all_loggers
    except ImportError:
        # Fallback to basic logging if pylogger not available
        def get_python_logger(level):
            logging.basicConfig(level=getattr(logging, level, logging.INFO))
        def force_reconfigure_all_loggers():
            pass


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
        # Import tools with fallback handling
        logging.info("Starting MCP tool registration...")
        try:
            from .tools.observability_vllm_tools import (
                list_models,
                list_namespaces,
                get_model_config,
                get_vllm_metrics_tool,
                analyze_vllm,
                calculate_metrics,
            )
        except ImportError:
            try:
                from tools.observability_vllm_tools import (
                    list_models,
                    list_namespaces,
                    get_model_config,
                    get_vllm_metrics_tool,
                    analyze_vllm,
                    calculate_metrics,
                )
            except ImportError:
                # Load tools directly via importlib
                tools_path = os.path.join(current_dir, 'tools', 'observability_vllm_tools.py')
                spec = importlib.util.spec_from_file_location("vllm_tools", tools_path)
                vllm_tools = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(vllm_tools)
                list_models = vllm_tools.list_models
                list_namespaces = vllm_tools.list_namespaces
                get_model_config = vllm_tools.get_model_config
                get_vllm_metrics_tool = vllm_tools.get_vllm_metrics_tool
                analyze_vllm = vllm_tools.analyze_vllm
                calculate_metrics = vllm_tools.calculate_metrics

        try:
            from .tools.observability_openshift_tools import (
                analyze_openshift,
                list_openshift_metric_groups,
                list_openshift_namespace_metric_groups,
            )
        except ImportError:
            try:
                from tools.observability_openshift_tools import (
                    analyze_openshift,
                    list_openshift_metric_groups,
                    list_openshift_namespace_metric_groups,
                )
            except ImportError:
                # Load tools directly via importlib
                tools_path = os.path.join(current_dir, 'tools', 'observability_openshift_tools.py')
                spec = importlib.util.spec_from_file_location("openshift_tools", tools_path)
                openshift_tools = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(openshift_tools)
                analyze_openshift = openshift_tools.analyze_openshift
                list_openshift_metric_groups = openshift_tools.list_openshift_metric_groups
                list_openshift_namespace_metric_groups = openshift_tools.list_openshift_namespace_metric_groups

        try:
            from .tools.prometheus_tools import (
                search_metrics,
                get_metric_metadata,
                get_label_values,
                execute_promql,
                explain_results,
                suggest_queries,
                select_best_metric,
            )
        except ImportError:
            try:
                from tools.prometheus_tools import (
                    search_metrics,
                    get_metric_metadata,
                    get_label_values,
                    execute_promql,
                    explain_results,
                    suggest_queries,
                    select_best_metric,
                )
            except ImportError:
                # Load tools directly via importlib
                tools_path = os.path.join(current_dir, 'tools', 'prometheus_tools.py')
                spec = importlib.util.spec_from_file_location("prometheus_tools", tools_path)
                prometheus_tools = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(prometheus_tools)
                search_metrics = prometheus_tools.search_metrics
                get_metric_metadata = prometheus_tools.get_metric_metadata
                get_label_values = prometheus_tools.get_label_values
                execute_promql = prometheus_tools.execute_promql
                explain_results = prometheus_tools.explain_results
                suggest_queries = prometheus_tools.suggest_queries
                select_best_metric = prometheus_tools.select_best_metric
                find_best_metric_with_metadata = prometheus_tools.find_best_metric_with_metadata
                find_best_metric_with_metadata_v2 = prometheus_tools.find_best_metric_with_metadata_v2

        # Register tools with error handling
        try:
            logging.info("Registering observability tools...")
            self.mcp.tool()(list_models)
            self.mcp.tool()(list_namespaces)
            self.mcp.tool()(get_model_config)
            self.mcp.tool()(get_vllm_metrics_tool)
            self.mcp.tool()(analyze_vllm)
            self.mcp.tool()(calculate_metrics)
            self.mcp.tool()(analyze_openshift)
            self.mcp.tool()(list_openshift_metric_groups)
            self.mcp.tool()(list_openshift_namespace_metric_groups)
            
            logging.info("Registering Prometheus tools...")
            self.mcp.tool()(search_metrics)
            self.mcp.tool()(get_metric_metadata)
            self.mcp.tool()(get_label_values)
            self.mcp.tool()(execute_promql)
            self.mcp.tool()(explain_results)
            self.mcp.tool()(suggest_queries)
            self.mcp.tool()(select_best_metric)
            self.mcp.tool()(find_best_metric_with_metadata)
            self.mcp.tool()(find_best_metric_with_metadata_v2)
            
            tool_count = len(self.mcp._tool_manager._tools) if hasattr(self.mcp, '_tool_manager') else 0
            logging.info(f"Successfully registered {tool_count} MCP tools")
        except Exception as e:
            logging.error(f"Failed to register MCP tools: {e}")
            import traceback
            traceback.print_exc()

