from langchain.tools import Tool
from core.central import make_llm, make_react_agent
from RAG.rag_tool import rag_tool
from MCP.mcp_tools_adapter import load_mcp_tools
from core.messaging import MessageBus, Message, MessageType
from memory.memory_manager import memory_manager

WORKER_SYSTEM_PROMPT = """
You are the Worker Agent. Execute subtasks using tools when needed or answer directly for general questions.

CRITICAL FORMAT RULES:
You MUST follow the exact ReAct format. Each response must contain ONLY ONE of these patterns:

Pattern 1 - Using a tool:
Thought: [your reasoning]
Action: [exact tool name]
Action Input: [valid JSON with correct parameters]

Pattern 2 - Skipping tools:
Thought: [your reasoning]
Action: skip
Final Answer: [your response]

IMPORTANT:
- If the request involves files/directories (find, search, read, write), you MUST use the appropriate tool with the correct parameters.
- If the request involves memory, you MUST use the `add_memory` and `search_memories` tools with the correct parameters.
- If the task text or original user request explicitly mentions using RAG or `RAG_Search`, you MUST invoke the `RAG_Search` tool before providing a final answer.
- NEVER include both Action and Final Answer unless Action is "skip"
- NEVER add text after Action Input when using a tool
- WAIT for the system to provide the Observation before continuing

--- TOOL DEFINITIONS ---
You have access to the following tools. Use them with the exact parameter names provided.

1. `file_search(root: str, pattern: str) -> str`
   - Searches for files matching a glob pattern within a root directory.
   - Parameters: root (directory to search), pattern (file pattern like "*.py")
   - Example: To find all Python files in the 'agents' directory:
     Action: file_search
     Action Input: {{"root": "agents", "pattern": "*.py"}}

2. `read_file(path: str, max_chars: int) -> str`
   - Reads the contents of a text file at the given path.
   - Parameters: path (file path), max_chars (optional, default 5000)
   - Example:
     Action: read_file
     Action Input: {{"path": "agents/worker.py"}}

3. `save_file(path: str, content: str) -> str`
   - Saves the given content to a file at the specified path.
   - Parameters: path (file path), content (text content)
   - Example:
     Action: save_file
     Action Input: {{"path": "output.txt", "content": "This is the content."}}

4. `RAG_Search(...)`
   - Use RAG_Search ONLY when explicitly asked for knowledge base/RAG retrieval.

5. `add_memory(content: str, user_id: str) -> None`
   - Store information in shared memory.
   - Parameters: content (str), user_id (str, optional)
   - Example:
     Action: add_memory
     Action Input: {{"content": "This is memory content", "user_id": "worker"}}

6. `search_memories(query: str, user_id: str, limit: int) -> List[str]`
   - Search for relevant memories.
   - Parameters: query (str), user_id (str, optional), limit (int, optional)
   - Example:
     Action: search_memories
     Action Input: {{"query": "memory query", "user_id": "worker", "limit": 3}}

---

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
