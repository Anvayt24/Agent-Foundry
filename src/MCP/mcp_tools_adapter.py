import sys
import asyncio
import json
import logging
from pathlib import Path
from langchain.tools import Tool
from pydantic import BaseModel, create_model
from typing import Type, Dict, Any

logger = logging.getLogger(__name__)

_CACHED_TOOLS = None

TYPE_MAP = {
    "string": str,
    "number": float,
    "integer": int,
    "boolean": bool,
    "object": dict,
    "array": list,
}


def _server_params():
    """Build the StdioServerParameters that spawn the bundled MCP server."""
    from mcp import StdioServerParameters

    server_path = str((Path(__file__).parent / "MCP_servers.py").resolve())
    return StdioServerParameters(command=sys.executable, args=[server_path])

def _build_argument_schema(model_name: str, json_schema: Dict[str, Any]) -> Type[BaseModel]:
    """
    Dynamically creates a Pydantic model from a JSON schema dictionary.

    This model is used by LangChain to understand the arguments for a tool.

    Args:
        model_name: The desired name for the created Pydantic model (e.g., "FileSearchArgs").
        json_schema: The JSON schema definition for the tool's parameters.

    Returns:
        A Pydantic BaseModel class representing the tool's arguments.
    """
    fields = {}
    properties = json_schema.get("properties", {})
    required_fields = json_schema.get("required", [])

    for prop_name, prop_schema in properties.items():
        field_type = TYPE_MAP.get(prop_schema.get("type"), Any)
        
        if "default" in prop_schema:
            fields[prop_name] = (field_type, prop_schema["default"])
        elif prop_name in required_fields:
            fields[prop_name] = (field_type, ...)  # Ellipsis marks a required field
        else:
            fields[prop_name] = (field_type, None) # Optional field
            
    return create_model(model_name, **fields)
    
def _create_mcp_tool_wrapper(tool_name: str, tool_description: str, tool_parameters: Dict[str, Any]):
    """Create a Tool wrapper for MCP tools with a proper args_schema."""
    ArgsModel = _build_argument_schema(tool_name, tool_parameters)

    def call_mcp_tool(*args, **kwargs) -> str:
        """Call MCP tool with the provided arguments."""
        try:
            if args and kwargs:
                raise ValueError("Mixing positional and keyword arguments is not supported for MCP tools")

            if args:
                if len(args) != 1:
                    raise ValueError("MCP tools accept at most one positional argument")
                arg = args[0]
                if isinstance(arg, BaseModel):
                    kwargs = arg.model_dump()
                elif isinstance(arg, dict):
                    kwargs = arg
                elif isinstance(arg, str):
                    try:
                        kwargs = json.loads(arg)
                    except json.JSONDecodeError:
                        kwargs = {"input": arg}
                else:
                    raise TypeError(f"Unsupported positional argument type: {type(arg).__name__}")

            async def _call():
                from mcp import ClientSession
                from mcp.client.stdio import stdio_client

                async with stdio_client(_server_params()) as (read, write):
                    async with ClientSession(read, write) as session:
                        await session.initialize()
                        # Pass the arguments received from the agent directly to the tool
                        result = await session.call_tool(tool_name, kwargs)

                        if hasattr(result, "content") and result.content:
                            if isinstance(result.content, list):
                                return "\n".join(str(item.text) if hasattr(item, "text") else str(item) for item in result.content)
                            return str(result.content)
                        return str(result)

            return asyncio.run(_call())

        except Exception as e:
            logger.exception("Error calling MCP tool %s", tool_name)
            return f"Error calling {tool_name}: {str(e)}"
    
    #langchain tool with the schema so the agent knows the arguments.
    return Tool(
        name=tool_name,
        func=call_mcp_tool,
        description=tool_description or f"MCP tool: {tool_name}",
        args_schema=ArgsModel,
    )

def load_mcp_tools():
    """Load MCP tools as structured Tool objects with self-described arguments."""
    global _CACHED_TOOLS
    if _CACHED_TOOLS is not None:
        return _CACHED_TOOLS

    try:
        from mcp import ClientSession
        from mcp.client.stdio import stdio_client

        async def _get_tool_info():
            async with stdio_client(_server_params()) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    tools_response = await session.list_tools()
                    # Fetch the tool's parameters (its schema) as well.
                    return [(tool.name, tool.description, tool.inputSchema) for tool in tools_response.tools]

        tool_info = asyncio.run(_get_tool_info())

        # Create a structured tool for each item returned by the server.
        tools = [_create_mcp_tool_wrapper(name, desc, params) for name, desc, params in tool_info]

        logger.info("Loaded %d MCP tool(s): %s", len(tools), ", ".join(t.name for t in tools))
        _CACHED_TOOLS = tools
        return tools
    except Exception:
        logger.exception("Could not load MCP tools")
        _CACHED_TOOLS = []
        return []