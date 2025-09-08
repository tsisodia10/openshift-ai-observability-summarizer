#!/usr/bin/env python3
"""Debug version of STDIO MCP Server to see what's wrong."""

import sys
import os

# Add the parent directory to Python path so we can import mcp_server
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def main():
    """Run the MCP server over STDIO with debug output."""
    try:
        print("DEBUG: Starting MCP server...", file=sys.stderr)
        
        # Set environment variables
        os.environ["FASTMCP_NO_BANNER"] = "1"
        os.environ["PYTHONUNBUFFERED"] = "1"
        
        print("DEBUG: Environment variables set", file=sys.stderr)
        
        # Import and initialize server
        print("DEBUG: Importing mcp_server.mcp...", file=sys.stderr)
        from mcp_server.mcp import ObservabilityMCPServer
        
        print("DEBUG: Creating server instance...", file=sys.stderr)
        server = ObservabilityMCPServer()
        
        print("DEBUG: Starting server.run()...", file=sys.stderr)
        # Run MCP server
        server.mcp.run(transport="stdio", show_banner=False)
        
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
