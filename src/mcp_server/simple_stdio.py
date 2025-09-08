#!/usr/bin/env python3
"""Simple STDIO MCP Server without naming conflicts."""

import sys
import os

# Add the parent directory to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def main():
    """Run the MCP server over STDIO."""
    try:
        # Set environment variables
        os.environ["FASTMCP_NO_BANNER"] = "1"
        os.environ["PYTHONUNBUFFERED"] = "1"
        
        # Import using absolute path to avoid conflicts
        import importlib.util
        mcp_file = os.path.join(os.path.dirname(__file__), "mcp.py")
        spec = importlib.util.spec_from_file_location("mcp_module", mcp_file)
        mcp_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mcp_module)
        
        # Initialize server
        server = mcp_module.ObservabilityMCPServer()
        
        # Run MCP server over STDIO
        server.mcp.run(transport="stdio", show_banner=False)
        
    except Exception as e:
        # Write error to stderr for debugging
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
