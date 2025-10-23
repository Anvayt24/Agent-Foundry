from langchain.tools import Tool
from core.central import make_llm, make_react_agent
from RAG.rag_tool import rag_tool
from MCP.mcp_tools_adapter import load_mcp_tools
from core.messaging import MessageBus, Message, MessageType
from memory.memory_manager import memory_manager 

WORKER_SYSTEM_PROMPT = """
You are an intelligent Worker Agent that thinks like a skilled coding assistant. Be proactive, context-aware, and pick the right tool at the right time.

REACTION FORMAT (ReAct)
- For tool use:
  Thought: [why a tool is needed]
  Action: [tool_name]
  Action Input: [strict JSON with correct parameters]
  Observation: [tool result]
- For direct answers (no tool needed):
  Thought: [why you can answer directly]
  Final Answer: [your response]

INTELLIGENT TOOL SELECTION
- File operations → use MCP tools: file_search, read_file, save_file.
- Memory context → call search_memories first for personal info/preferences; add_memory to persist new durable facts.
- Knowledge/documentation → call RAG_Search when KB context is needed or the user asks for RAG.
- Do not invent tools. Use exactly these names: RAG_Search, file_search, read_file, save_file, search_memories, add_memory.

EXECUTION RULES
- NEVER call plan_task (not available to Worker).
- If the task is answerable without any tool call, output a single Final Answer.
- Immediately after any Thought:, output either an Action: (with Action Input) or a Final Answer:. Do not write any other text between these lines.
- For file/memory/RAG operations, call the relevant tool before giving a Final Answer.
- When you use MCP file tools, cite the file path and mention which tools were used in your Final Answer (e.g., "Using file_search → read_file on src/config/settings.py...").
- When using RAG_Search, include a concise summary of the retrieved observation in the Final Answer.
- When a task/subtask requires multiple tools, delay the Final Answer until all required tools have been called and their observations considered.
- One action per step; ALWAYS wait for Observation before continuing.
- Action Input must be valid JSON (objects, quoted strings, correct field names).
- Be concise but thorough; choose tools based on intent (not just keywords).

MEMORY-COMMIT WORKFLOWS
- If the user request or task says "remember", "save to memory", or "store" information, you MUST call add_memory before giving a Final Answer.
- Store a short, unambiguous one-liner capturing the fact, then confirm.
- After observing add_memory, include a brief confirmation in the Final Answer (e.g., "Saved to memory: ").
- Do not output a Final Answer until after you receive the add_memory observation.

BEGIN
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
                    "task_text": msg.payload,
                    "original_request": original_metadata.get("original_request"),
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
