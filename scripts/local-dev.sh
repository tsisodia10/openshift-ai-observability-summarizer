#!/bin/bash

# AI Observability Metric Summarizer - Local Development Script

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
PROMETHEUS_NAMESPACE="openshift-monitoring"
METRIC_API_APP="metrics-api-app"
THANOS_PORT=9090
LLAMASTACK_PORT=8321
LLAMA_MODEL_PORT=8080
# Metrics API (FastAPI) port for local dev; can override via METRICS_API_PORT
METRICS_API_PORT=${METRICS_API_PORT:-8000}
UI_PORT=8501
MCP_PORT=${MCP_PORT:-8085}

echo -e "${BLUE}🚀 AI Observability Metric Summarizer - Local Development Setup${NC}"
echo "=============================================================="

# Function to display usage
usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -n/-N NAMESPACE              Default namespace for pods (required)"
    echo "  -m/-M NAMESPACE              Llama Model namespace (optional, use if model is in different namespace)"
    echo ""
    echo "Examples:"
    echo "  $0 -n default-ns                       # All pods/services in same namespace"
    echo "  $0 -N default-ns                       # All pods/services in same namespace (uppercase)"
    echo "  $0 -n default-ns -m model-ns           # Model in different namespace than other pods/services"
}

# Function to parse command line arguments
parse_args() {
    # Check if no arguments provided
    if [ $# -eq 0 ]; then
        usage
        exit 2
    fi

    DEFAULT_NAMESPACE=""
    LLAMA_MODEL_NAMESPACE=""

    while getopts "n:N:m:M:" opt; do
        case $opt in
            n|N) DEFAULT_NAMESPACE="$OPTARG"
                 ;;
            m|M) LLAMA_MODEL_NAMESPACE="$OPTARG"
                 ;;
            *) echo -e "${RED}❌ INVALID option: [$OPTARG]${NC}"
               usage
               exit 1
               ;;
        esac
    done

    # Validate arguments
    if [ -z "$DEFAULT_NAMESPACE" ]; then
        echo -e "${RED}❌ Default namespace is required. Please specify using -n or -N${NC}"
        usage
        exit 1
    fi

    # Set llama model namespace to default if not provided
    if [ -z "$LLAMA_MODEL_NAMESPACE" ]; then
        LLAMA_MODEL_NAMESPACE="$DEFAULT_NAMESPACE"
    fi
}

# Function to cleanup on exit
cleanup() {
    # Prevent multiple cleanup calls
    if [ "$CLEANUP_DONE" = "true" ]; then
        return
    fi
    CLEANUP_DONE=true

    echo -e "\n${YELLOW}🧹 Cleaning up services and port-forwards...${NC}"
    ensure_port_free "$METRICS_API_PORT"
    ensure_port_free "$MCP_PORT"
    pkill -f "oc port-forward" || true
    pkill -f "uvicorn.*metrics_api:app" || true
    pkill -f "mcp_server.main" || true
    pkill -f "streamlit run ui.py" || true

    # Deactivate virtual environment if it was activated
    if [ -n "$VIRTUAL_ENV" ]; then
        echo -e "${BLUE}🐍 Deactivating virtual environment...${NC}"
        deactivate
    fi

    echo -e "${GREEN}✅ Cleanup complete${NC}"
}

# Function to check prerequisites and activate virtual environment
check_prerequisites() {
    echo -e "${BLUE}🔍 Checking prerequisites...${NC}"

    # Check for virtual environment and activate it
    if [ -f ".venv/bin/activate" ]; then
        echo -e "${BLUE}🐍 Activating Python virtual environment...${NC}"
        source .venv/bin/activate
        echo -e "${GREEN}✅ Virtual environment activated${NC}"
    else
        echo -e "${YELLOW}⚠️  Virtual environment (.venv) not found${NC}"
        echo -e "${YELLOW}   Please create virtual environment by following README or DEV_GUIDE${NC}"
        exit 1
    fi

    if ! command -v oc &> /dev/null; then
        echo -e "${RED}❌ OpenShift CLI (oc) is not installed${NC}"
        exit 1
    fi

    if ! oc whoami &> /dev/null; then
        echo -e "${RED}❌ Not logged in to OpenShift cluster${NC}"
        echo -e "${YELLOW}   Please run: oc login${NC}"
        exit 1
    fi

    echo -e "${GREEN}✅ Prerequisites check passed${NC}"
}

# Function to find and start port forwards
start_port_forwards() {
    echo -e "${BLUE}🔍 Finding pods and starting port-forwards...${NC}"
    
    # Find Thanos pod
    THANOS_POD=$(oc get pods -n "$PROMETHEUS_NAMESPACE" -o name | grep thanos-querier | head -1 | cut -d'/' -f2 || echo "")
    if [ -z "$THANOS_POD" ]; then
        THANOS_POD=$(oc get pods -n "$PROMETHEUS_NAMESPACE" -o name | grep prometheus | head -1 | cut -d'/' -f2 || echo "")
    fi
    
    if [ -n "$THANOS_POD" ]; then
        echo -e "${GREEN}✅ Found Thanos pod: $THANOS_POD${NC}"
        oc port-forward pod/"$THANOS_POD" "$THANOS_PORT:9090" -n "$PROMETHEUS_NAMESPACE" &
        echo -e "${GREEN}   📊 Thanos available at: http://localhost:$THANOS_PORT${NC}"
    else
        echo -e "${RED}❌ No Thanos/Prometheus pod found${NC}"
        exit 1
    fi
    
    # Find LlamaStack pod
    LLAMASTACK_POD=$(oc get pods -n "$DEFAULT_NAMESPACE" -o name | grep -E "(llama-stack|llamastack)" | head -1 | cut -d'/' -f2 || echo "")
    if [ -n "$LLAMASTACK_POD" ]; then
        echo -e "${GREEN}✅ Found LlamaStack pod: $LLAMASTACK_POD${NC}"
        oc port-forward pod/"$LLAMASTACK_POD" "$LLAMASTACK_PORT:8321" -n "$DEFAULT_NAMESPACE" &
        echo -e "${GREEN}   🦙 LlamaStack available at: http://localhost:$LLAMASTACK_PORT${NC}"
    else
        echo -e "${RED}❌  LlamaStack pod not found. Exiting...${NC}"
        exit 1
    fi
    
    # Find Llama Model service
    LLAMA_MODEL_SERVICE=$(oc get services -n "$LLAMA_MODEL_NAMESPACE" -o name | grep -E "(llama-3|predictor)" | grep -v stack | head -1 | cut -d'/' -f2 || echo "")
    if [ -n "$LLAMA_MODEL_SERVICE" ]; then
        echo -e "${GREEN}✅ Found Llama Model service: $LLAMA_MODEL_SERVICE in [$LLAMA_MODEL_NAMESPACE] namespace${NC}"
        oc port-forward service/"$LLAMA_MODEL_SERVICE" "$LLAMA_MODEL_PORT:8080" -n "$LLAMA_MODEL_NAMESPACE" &
        echo -e "${GREEN}   🤖 Llama Model available at: http://localhost:$LLAMA_MODEL_PORT${NC}"
    else
        echo -e "${RED}❌  Llama Model service not found in namespace: $LLAMA_MODEL_NAMESPACE. Exiting...${NC}"
        exit 1
    fi
    
    sleep 3  # Give port-forwards time to establish
}

# Ensure a TCP port is free by terminating any process listening on it
ensure_port_free() {
    local port=$1
    if lsof -nP -iTCP:"$port" -sTCP:LISTEN >/dev/null 2>&1; then
        echo -e "${YELLOW}⚠️  Port $port is in use. Attempting to free it...${NC}"
        # Try graceful termination first
        lsof -nP -iTCP:"$port" -sTCP:LISTEN -t | xargs -r kill || true
        sleep 1
        # Force kill if still listening
        if lsof -nP -iTCP:"$port" -sTCP:LISTEN >/dev/null 2>&1; then
            lsof -nP -iTCP:"$port" -sTCP:LISTEN -t | xargs -r kill -9 || true
        fi
        if lsof -nP -iTCP:"$port" -sTCP:LISTEN >/dev/null 2>&1; then
            echo -e "${RED}❌ Could not free port $port. Please free it and retry.${NC}"
            exit 1
        fi
        echo -e "${GREEN}✅ Port $port is now free${NC}"
    fi
}

# This function sets "MODEL_CONFIG" environment variable from cluster deployment or local file
set_model_config() {
    echo -e "${BLUE}🔧 Setting up MODEL_CONFIG...${NC}"
    
    # Check if local model-config.json exists
    LOCAL_MODEL_CONFIG="deploy/helm/model-config.json"
    if [ -f "$LOCAL_MODEL_CONFIG" ]; then
        echo -e "${YELLOW}📋 Found local model config file: $LOCAL_MODEL_CONFIG${NC}"
        echo -e "${YELLOW}   This includes additional models like Anthropic Claude for testing.${NC}"
        echo ""
        echo -e "${BLUE}Choose MODEL_CONFIG source:${NC}"
        echo -e "  ${GREEN}1) Use CLUSTER config${NC} (production models from deployed metrics-api)"
        echo -e "  ${YELLOW}2) Use LOCAL config${NC} (includes Anthropic/Claude and other test models)"
        echo ""
        read -p "Enter your choice (1 or 2) [default: 1]: " CONFIG_CHOICE
        
        if [ "$CONFIG_CHOICE" = "2" ]; then
            echo -e "${YELLOW}🔧 Using LOCAL model config...${NC}"
            export MODEL_CONFIG=$(cat "$LOCAL_MODEL_CONFIG")
            if [ -n "$MODEL_CONFIG" ]; then
                echo -e "${GREEN}✅ LOCAL MODEL_CONFIG loaded successfully${NC}"
                echo -e "${BLUE}   Available models: $(echo "$MODEL_CONFIG" | jq -r 'keys | join(", ")')${NC}"
                return 0
            else
                echo -e "${RED}❌ Failed to read local model config file${NC}"
                exit 1
            fi
        fi
    fi
    
    # Default: Use cluster config
    echo -e "${BLUE}🔧 Using CLUSTER model config...${NC}"
    METRIC_API_DEPLOYMENT=$(oc get deploy "$METRIC_API_APP" -n "$DEFAULT_NAMESPACE")
    if [ -n "$METRIC_API_DEPLOYMENT" ]; then
        echo -e "${GREEN}✅ Found [$METRIC_API_APP] deployment: $METRIC_API_DEPLOYMENT${NC}"
        export $(oc set env deployment/$METRIC_API_APP --list  -n "$DEFAULT_NAMESPACE" | grep MODEL_CONFIG)
        if [ -n "$MODEL_CONFIG" ]; then
          echo -e "${GREEN}✅ CLUSTER MODEL_CONFIG set successfully${NC}"
        else
          echo -e "${RED}❌ Unable to set MODEL_CONFIG environment variable. It is required to run the UI locally.${NC}"
          exit 1
        fi
    else
        echo -e "${RED}❌ $METRIC_API_APP deployment not found. It is required to set MODEL_CONFIG.${NC}"
        exit 1
    fi
}

# Function to start local services
start_local_services() {
    echo -e "${BLUE}🏃 Starting local services...${NC}"
    
    # Get service account token
    TOKEN=$(oc whoami -t)
    
    # Set environment variables
    export PROMETHEUS_URL="http://localhost:$THANOS_PORT"
    export LLAMA_STACK_URL="http://localhost:$LLAMASTACK_PORT/v1/openai/v1"
    export THANOS_TOKEN="$TOKEN"
    export METRICS_API_URL="http://localhost:$METRICS_API_PORT"
    export MCP_URL="http://localhost:$MCP_PORT"
    export PROM_URL="http://localhost:$THANOS_PORT"
    # Set log level (override with PYTHON_LOG_LEVEL=DEBUG for more verbose logging)
    export PYTHON_LOG_LEVEL="${PYTHON_LOG_LEVEL:-INFO}"

    # macOS weasyprint support
    export DYLD_FALLBACK_LIBRARY_PATH="/opt/homebrew/lib:$DYLD_FALLBACK_LIBRARY_PATH"

    set_model_config
    
    # Start Metrics API backend
    echo -e "${BLUE}🔧 Starting Metrics API backend...${NC}"
    ensure_port_free "$METRICS_API_PORT"
    (cd src/api && python3 -m uvicorn metrics_api:app --host 0.0.0.0 --port $METRICS_API_PORT --reload > log.txt) &
    MCP_PID=$!
    
    # Wait for Metrics API to start
    sleep 3
    
    # Test Metrics API service
    if curl -s --connect-timeout 5 "http://localhost:$METRICS_API_PORT/models" > /dev/null; then
        echo -e "${GREEN}✅ Metrics API backend started successfully${NC}"
    else
        echo -e "${RED}❌ Metrics API backend failed to start${NC}"
        exit 1
    fi

    # Start MCP server (HTTP transport)
    echo -e "${BLUE}🧩 Starting MCP Server (HTTP)...${NC}"
    ensure_port_free "$MCP_PORT"
    (cd src && \
      MCP_TRANSPORT_PROTOCOL=http \
      MODEL_CONFIG="$MODEL_CONFIG" \
      PROMETHEUS_URL="$PROMETHEUS_URL" \
      LLAMA_STACK_URL="$LLAMA_STACK_URL" \
      THANOS_TOKEN="$THANOS_TOKEN" \
      python3 -m mcp_server.main > mcp_log.txt) &
    MCP_SRV_PID=$!

    # Wait for MCP server to start
    sleep 3

    # Test MCP server health
    if curl -s --connect-timeout 5 "http://localhost:$MCP_PORT/health" | grep -q '"status"'; then
        echo -e "${GREEN}✅ MCP Server started successfully on port $MCP_PORT${NC}"
    else
        echo -e "${RED}❌ MCP Server failed to start${NC}"
        exit 1
    fi
    
    # Start Streamlit UI
    echo -e "${BLUE}🎨 Starting Streamlit UI...${NC}"
    (cd src/ui && \
      MCP_SERVER_URL="http://localhost:$MCP_PORT" \
      streamlit run ui.py --server.port $UI_PORT --server.address 0.0.0.0 --server.headless true) &
    UI_PID=$!
    
    # Wait for UI to start
    sleep 5
    
    echo -e "${GREEN}✅ All services started successfully!${NC}"
}

# Main execution
main() {
    parse_args "$@"
    check_prerequisites

    # Set cleanup trap only after successful prerequisite checks
    trap cleanup EXIT INT TERM

    echo ""
    echo -e "${BLUE}--------------------------------${NC}"
    echo -e "${BLUE}Namespaces being used for setup:${NC}"
    echo -e "${BLUE}  DEFAULT_NAMESPACE: $DEFAULT_NAMESPACE${NC}"
    echo -e "${BLUE}  LLAMA_MODEL_NAMESPACE: $LLAMA_MODEL_NAMESPACE${NC}"
    echo -e "${BLUE}--------------------------------${NC}\n"

    start_port_forwards
    start_local_services
    
    echo -e "\n${GREEN}🎉 Setup complete! All services are running.${NC}"
    echo -e "\n${BLUE}📋 Services Available:${NC}"
    echo -e "   ${YELLOW}🎨 Streamlit UI: http://localhost:$UI_PORT${NC}"
    echo -e "   ${YELLOW}🔧 Metrics API: http://localhost:$METRICS_API_PORT/docs${NC}"
    echo -e "   ${YELLOW}🧩 MCP Server (health): http://localhost:$MCP_PORT/health${NC}"
    echo -e "   ${YELLOW}🧩 MCP HTTP Endpoint: http://localhost:$MCP_PORT/mcp${NC}"
    echo -e "   ${YELLOW}📊 Prometheus: http://localhost:$THANOS_PORT${NC}"
    echo -e "   ${YELLOW}🦙 LlamaStack: http://localhost:$LLAMASTACK_PORT${NC}"
    echo -e "   ${YELLOW}🤖 Llama Model: http://localhost:$LLAMA_MODEL_PORT${NC}"
    
    echo -e "\n${GREEN}🎯 Ready to use! Open your browser to http://localhost:$UI_PORT${NC}"
    echo -e "\n${YELLOW}📝 Note: Keep this terminal open to maintain all services${NC}"
    echo -e "${YELLOW}Press Ctrl+C to stop all services and cleanup${NC}"
    
    # Keep script running
    wait
}

# Run main function
main "$@"