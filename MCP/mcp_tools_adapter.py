import os
import sys
import asyncio
import json
from pathlib import Path
from langchain.tools import Tool

_CACHED_TOOLS = None

def _create_mcp_tool_wrapper(tool_name: str, tool_description: str):
    """Create a simple Tool wrapper for MCP tools."""
    
    def call_mcp_tool(input_str: str) -> str:
        """Call MCP tool with JSON string input."""
        try:
            # Parse JSON input
            if input_str.strip().startswith("{"):
                args = json.loads(input_str)
            else:
                # Fallback: treat as simple query
                args = {"query": input_str}
            
            # Call MCP server
            async def _call():
                from mcp import ClientSession, StdioServerParameters
                from mcp.client.stdio import stdio_client
                
                server_path = str((Path(__file__).parent / "MCP_servers.py").resolve())
                params = StdioServerParameters(command=sys.executable, args=[server_path])
                
                async with stdio_client(params) as (read, write):
                    async with ClientSession(read, write) as session:
                        await session.initialize()
                        result = await session.call_tool(tool_name, args)
                        
                        if hasattr(result, "content") and result.content:
                            if isinstance(result.content, list):
                                return "\n".join(str(item.text) if hasattr(item, "text") else str(item) for item in result.content)
                            else:
                                return str(result.content)
                        return str(result)
            
            return asyncio.run(_call())
            
        except Exception as e:
            return f"Error calling {tool_name}: {str(e)}"
    
    return Tool(
        name=tool_name,
        func=call_mcp_tool,
        description=tool_description or f"MCP tool: {tool_name}"
    )

def load_mcp_tools():
    """Load MCP tools as simple Tool objects."""
    global _CACHED_TOOLS
    if _CACHED_TOOLS is not None:
        return _CACHED_TOOLS

    try:
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client

        async def _get_tool_info():
            server_path = str((Path(__file__).parent / "MCP_servers.py").resolve())
            params = StdioServerParameters(command=sys.executable, args=[server_path])

            async with stdio_client(params) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    tools_response = await session.list_tools()
                    return [(tool.name, tool.description) for tool in tools_response.tools]

        tool_info = asyncio.run(_get_tool_info())
        tools = [_create_mcp_tool_wrapper(name, desc) for name, desc in tool_info]
        
        _CACHED_TOOLS = tools
        return tools
    except Exception as e:
        print(f"[MCP] Error: {e}")
        _CACHED_TOOLS = []
        return []
