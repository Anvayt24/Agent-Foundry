from langchain.tools import Tool
from central import make_llm, make_react_agent
from RAG.rag_tool import rag_tool
from MCP.mcp_tools_adapter import load_mcp_tools
from messaging import MessageBus, Message, MessageType

WORKER_SYSTEM_PROMPT = """
You are a Worker Agent.
You execute the subtask assigned to you using the tools provided.
Always follow the ReAct format:

Thought: reasoning
Action: tool_name
Action Input: the input
Observation: tool output
Final Answer: the final completed subtask output
"""

class WorkerA2A:
    def __init__(self, message_bus: MessageBus):
        self.message_bus = message_bus
        self.message_bus.register_agent("Worker")

    def _make_agent(self):
        rag_tool_wrapper = Tool(
            name="RAG_Search",
            func=rag_tool,
            description="Search the document knowledge base for relevant information",
        )
        tools = [rag_tool_wrapper]
        tools += load_mcp_tools()
        return make_react_agent(
            tools=tools,
            llm=make_llm(temp=0),
            system_prompt=WORKER_SYSTEM_PROMPT,
        )

    def perform_task(self, task_data: str) -> str:
        agent = self._make_agent()
        try:
            out = agent.invoke({"input": task_data})
            return out.get("output", str(out))
        except Exception as e:
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
        description="Search the document knowledge base for relevant information",
    )
    tools = [rag_tool_wrapper]
    tools += load_mcp_tools()
    return make_react_agent(
        tools=tools,
        llm=make_llm(temp=0),
        system_prompt=WORKER_SYSTEM_PROMPT,
    )


def create_worker_a2a(message_bus: MessageBus) -> WorkerA2A:
    return WorkerA2A(message_bus)
