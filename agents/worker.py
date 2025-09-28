from langchain.tools import Tool
from central import make_llm, make_react_agent
from RAG.rag_tool import rag_tool
from MCP.mcp_tools_adapter import load_mcp_tools
from messaging import MessageBus, Message, MessageType

WORKER_SYSTEM_PROMPT = """
You are the Worker Agent. Execute subtasks using tools when needed or answer directly for general questions.

CRITICAL FORMAT RULES:
You MUST follow the exact ReAct format. Each response must contain ONLY ONE of these patterns:

Pattern 1 - Using a tool:
Thought: [your reasoning]
Action: [exact tool name]
Action Input: [valid JSON parameters]

Pattern 2 - Skipping tools:
Thought: [your reasoning]
Action: skip
Final Answer: [your response]

IMPORTANT: 
- If the request involves files/directories (find, search, read, write), you MUST use the appropriate tool
- NEVER include both Action and Final Answer unless Action is "skip"
- NEVER add text after Action Input when using a tool
- WAIT for the system to provide the Observation before continuing

Tool selection:
- Use RAG_Search ONLY when explicitly asked for knowledge base/RAG retrieval
- Use file_search for finding files (e.g., "find all .py files in agents directory")
- Use read_file for reading file contents
- Use save_file for creating/writing files
- Use Action: skip ONLY for conceptual questions that don't require tools

Available tools: {tool_names}
"""

class WorkerA2A:
    def __init__(self, message_bus: MessageBus):
        self.message_bus = message_bus
        self.message_bus.register_agent("Worker")

    def _make_agent(self):
        rag_tool_wrapper = Tool(
            name="RAG_Search",
            func=rag_tool,
            description=(
                "Retrieve information from the project knowledge base. Use ONLY when the "
                "task explicitly requires KB retrieval (mentions 'RAG' or needs doc context)."
            ),
        )
        tools = [rag_tool_wrapper]
        mcp_tools = load_mcp_tools()
        tools += mcp_tools
        
        # Get tool names for the prompt
        tool_names = [tool.name for tool in tools]
        
        # Format the system prompt with tool names
        formatted_prompt = WORKER_SYSTEM_PROMPT.format(tool_names=", ".join(tool_names))
        
        return make_react_agent(
            tools=tools,
            llm=make_llm(temp=0.0),  # Lower temperature for more consistent output
            system_prompt=formatted_prompt,
        )

    def perform_task(self, task_data: str) -> str:
        agent = self._make_agent()
        try:
            out = agent.invoke({"input": task_data})
            return out.get("output", str(out))
        except Exception as e:
            # Handle common LangChain parsing errors
            error_msg = str(e)
            if "OUTPUT_PARSING_FAILURE" in error_msg or "both a final answer and a parse-able action" in error_msg:
                return f"Agent format error: The response contained both an action and final answer. Please retry with a simpler request."
            elif "Invalid or incomplete response" in error_msg:
                return f"Agent parsing error: The response format was invalid. Please retry."
            else:
                return f"Worker error: {e}"

    def process_once(self, timeout: float = 0.2) -> bool:
        msg = self.message_bus.receive("Worker", timeout=timeout)
        if not msg:
            return False
        if msg.message_type == MessageType.TASK_REQUEST:
            result = self.perform_task(msg.payload)
            resp = Message(
                sender="Worker",
                recipient="Verifier",
                message_type=MessageType.TASK_RESPONSE,
                payload=result,
                metadata={"task_id": (msg.metadata or {}).get("task_id")},
            )
            self.message_bus.send(resp)
        return True


def create_worker():
    """Orchestrator mode: return a LangChain worker agent executor."""
    rag_tool_wrapper = Tool(
        name="RAG_Search",
        func=rag_tool,
        description=(
            "Retrieve information from the project knowledge base. Use ONLY when the "
            "task explicitly requires KB retrieval (mentions 'RAG' or needs doc context)."
        ),
    )
    tools = [rag_tool_wrapper]
    mcp_tools = load_mcp_tools()
    tools += mcp_tools
    
    # Tools loaded successfully
    
    return make_react_agent(
        tools=tools,
        llm=make_llm(temp=0.0),  # Lower temperature for more consistent output
        system_prompt=WORKER_SYSTEM_PROMPT,
    )


def create_worker_a2a(message_bus: MessageBus) -> WorkerA2A:
    return WorkerA2A(message_bus)
