#!/usr/bin/env python3
"""Working STDIO MCP Server that avoids naming conflicts."""

import sys
import os
from pathlib import Path

# Ensure the project root is in the Python path
project_root = Path(__file__).resolve().parents[2]  # Go up 3 levels from this file
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Ensure the mcp_server directory is in the Python path
mcp_server_dir = Path(__file__).resolve().parent
if str(mcp_server_dir) not in sys.path:
    sys.path.insert(0, str(mcp_server_dir))

# Remove current directory from path to avoid mcp.py conflict
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir in sys.path:
    sys.path.remove(current_dir)

def main():
    """Run the MCP server over STDIO."""
    try:
        # Set environment variables
        os.environ["FASTMCP_NO_BANNER"] = "1"
        os.environ["PYTHONUNBUFFERED"] = "1"
        
        # Import and initialize server using absolute import
        from mcp_server.mcp import ObservabilityMCPServer
        server = ObservabilityMCPServer()
        
        # Run MCP server over STDIO
        server.mcp.run(transport="stdio", show_banner=False)
        
    except Exception as e:
        # Write error to stderr for debugging
        print(f"Error in working_stdio.py: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
