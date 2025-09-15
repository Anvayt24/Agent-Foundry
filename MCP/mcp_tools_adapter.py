import os

def load_mcp_tools():
    """
    Try to load tools from a running FastMCP server.
    If the optional client is missing or server is offline, return [].
    """
    mcp_url = os.getenv("MCP_SERVER_URL", "http://127.0.0.1:8000")
    try:
        # Optional dependency: pip install langchain-mcp
        from langchain_mcp import MCPToolkit
        toolkit = MCPToolkit.from_http(mcp_url)
        tools = toolkit.get_tools()
        # Tools are already LangChain Tool instances
        return tools
    except Exception as e:
        # Silent, graceful fallback (no MCP tools)
        print(f"[MCP] Skipping MCP tools ({e}). Using only built-in tools.")
        return []
