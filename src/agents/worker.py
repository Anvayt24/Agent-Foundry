from langchain.tools import Tool
from core.central import make_llm, make_react_agent
from RAG.rag_tool import rag_tool
from MCP.mcp_tools_adapter import load_mcp_tools
from core.messaging import MessageBus, Message, MessageType
from memory.memory_manager import memory_manager

WORKER_SYSTEM_PROMPT = """
You are an intelligent Worker Agent that thinks like a skilled coding assistant. Be proactive, context-aware, and choose the right tools automatically.

CRITICAL FORMAT RULES:
You MUST follow the exact ReAct format. Each response must contain ONLY ONE of these patterns:

Pattern 1 - Using a tool:
Thought: [your reasoning about why you need this tool]
Action: [exact tool name]
Action Input: [valid JSON with correct parameters]

Pattern 2 - Skipping tools (only for general knowledge questions):
Thought: [why you can answer without tools]
Action: skip
Final Answer: [your response]

INTELLIGENT TOOL SELECTION - Think like Windsurf/Cascade:
- **File Operations**: If someone asks "search for X file", "find X", "where is X file", "read X file" - IMMEDIATELY use file_search, read_file, or save_file. Don't plan - just do it.
- **Memory Operations**: For personal info queries ("my dog's name", "where does my friend live"), ALWAYS search_memories first to get context.
- **Auto-Save Memories**: When someone shares personal info ("I am an AI engineer", "my dog is named X", "my friend lives in Y"), AUTOMATICALLY use add_memory to save it for future reference.
- **RAG/Knowledge**: If the task needs project documentation or knowledge base info, use RAG_Search.
- **General Questions**: Only skip tools for purely theoretical questions that don't need data.

CONTEXT UNDERSTANDING:
- "User" in questions refers to the person asking (use first person context)
- "My X" and "user's X" mean the same thing - the person's X
- When searching files, start from current directory "." unless specified otherwise
- Be direct and actionable - don't over-explain or create unnecessary subtasks

EXECUTION RULES:
- NEVER use plan_task for simple, direct actions like file searches
- NEVER provide Final Answer for file/memory operations without calling the relevant tool first
- ALWAYS wait for Observation before continuing
- Be concise but thorough in your reasoning
- Choose tools based on intent, not literal keywords

Begin!
"""


class WorkerA2A:
    def __init__(self, message_bus: MessageBus):
        self.message_bus = message_bus
        self.message_bus.register_agent("Worker")

    def perform_task(self, task_data: str) -> str:
        agent = create_worker()
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

    def process_once(self, session_id: str | None = None, timeout: float = 0.2) -> bool:
        msg = self.message_bus.receive("Worker", timeout=timeout)
        if not msg:
            return False
        if session_id is not None:
            msg_session = (msg.metadata or {}).get("session_id")
            if msg_session != session_id:
                # Drop stale message belonging to a different session
                return False
        if msg.message_type == MessageType.TASK_REQUEST:
            result = self.perform_task(msg.payload)
            original_metadata = msg.metadata or {}
            resp = Message(
                sender="Worker",
                recipient="Verifier",
                message_type=MessageType.TASK_RESPONSE,
                payload=result,
                metadata={
                    "task_id": original_metadata.get("task_id"),
                    "session_id": original_metadata.get("session_id", session_id),
                },
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
    memory_tools = [
            Tool(
                name="add_memory",
                func=lambda content, user_id="agent_system": memory_manager.add_memory(content, user_id),
                description="Store information in shared memory. Input: content (str), user_id (str, optional)"
            ),
            Tool(
                name="search_memories",
                func=lambda query, user_id="agent_system", limit=5: memory_manager.search_memories(query, user_id, limit),
                description="Search for relevant memories. Input: query (str), user_id (str, optional), limit (int, optional)"
            )
        ]
    tools = [rag_tool_wrapper]
    mcp_tools = load_mcp_tools()
    tools += mcp_tools + memory_tools
    
    # Tools loaded successfully
    
    return make_react_agent(
        tools=tools,
        llm=make_llm(temp=0.0),  # Lower temperature for more consistent output
        system_prompt=WORKER_SYSTEM_PROMPT,
    )


def create_worker_a2a(message_bus: MessageBus) -> WorkerA2A:
    return WorkerA2A(message_bus)
