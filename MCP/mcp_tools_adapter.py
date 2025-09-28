import os
import sys
import asyncio
from pathlib import Path

# Cache to avoid reconnecting/spawning the MCP server repeatedly
_CACHED_TOOLS = None


def load_mcp_tools():
    """
    Load MCP tools from a FastMCP server using the official langchain-mcp-adapters.

    Supports two transports controlled by env vars:
      - MCP_TRANSPORT=stdio (default): spawns the local MCP server over stdio
        - MCP_SERVER_PATH: optional, path to MCP_servers.py (defaults to this folder)
      - MCP_TRANSPORT=streamable_http: connects to an HTTP MCP endpoint
        - MCP_SERVER_URL: e.g. http://127.0.0.1:8000/mcp/

    Returns a list of LangChain Tool instances or [] on failure.
    """
    global _CACHED_TOOLS
    if _CACHED_TOOLS is not None:
        return _CACHED_TOOLS

    try:
        # Imports kept inside to avoid hard dependency when unused
        from mcp import ClientSession
        from langchain_mcp_adapters.tools import load_mcp_tools as _load_mcp_tools

        transport = os.getenv("MCP_TRANSPORT", "stdio").lower()

        async def _load_stdio():
            from mcp import StdioServerParameters
            from mcp.client.stdio import stdio_client

            server_path = os.getenv(
                "MCP_SERVER_PATH",
                str((Path(__file__).parent / "MCP_servers.py").resolve()),
            )
            params = StdioServerParameters(command=sys.executable, args=[server_path])

            async with stdio_client(params) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    return await _load_mcp_tools(session)

        async def _load_http():
            from mcp.client.streamable_http import streamablehttp_client

            url = os.getenv("MCP_SERVER_URL", "http://127.0.0.1:8000/mcp/")
            async with streamablehttp_client(url) as (read, write, _):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    return await _load_mcp_tools(session)

        async def _run():
            if transport == "stdio":
                return await _load_stdio()
            elif transport in {"http", "streamable_http", "streamable-http"}:
                return await _load_http()
            else:
                raise ValueError(f"Unsupported MCP_TRANSPORT: {transport}")

        tools = asyncio.run(_run()) or []
        _CACHED_TOOLS = tools
        return tools
    except Exception as e:
        print(f"[MCP] Skipping MCP tools ({e}). Using only built-in tools.")
        _CACHED_TOOLS = []
        return []
