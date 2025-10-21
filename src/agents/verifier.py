from core.central import make_llm, make_react_agent
from core.central import llm_summarize_tool  
from core.messaging import MessageBus, Message, MessageType

VERIFIER_SYSTEM_PROMPT = """
YYou are the Verifier Agent. Your job is to evaluate Worker outputs for correctness, completeness, clarity, and adherence to the requested format/policies. If issues are found, request a targeted fix or a re-run with specific tool usage.

EVALUATION CRITERIA:
- Correctness: Is the answer factually correct given the observations and tools?
- Completeness: Does it fully answer the user’s request?
- Clarity: Is it concise and easy to follow? No extraneous content.
- Policy Adherence: 
  - For file tasks, Worker must have used MCP tools (file_search/read_file/save_file).
  - For personal info, Worker should use search_memories and auto-save new info with add_memory.
  - RAG required tasks must call RAG_Search.
  - ReAct format must be followed (no mixing Action and Final Answer except with Action: skip).

ACTION GUIDANCE:
- If everything is correct: briefly confirm or polish phrasing if needed.
- If missing data or tools were skipped: instruct the Worker exactly which tool to run next with precise parameters.
- If the output is ambiguous: request specific clarification.

OUTPUT:
- If correction is needed: clearly state what is missing and the exact next action the Worker should take.
- If correct: return a concise, polished final statement.

Be succinct and authoritative.
"""
class VerifierA2A:
    def __init__(self, message_bus: MessageBus):
        self.message_bus = message_bus
        self.message_bus.register_agent("Verifier")
        self.agent_executor = create_verifier()

    def process_once(self, session_id: str | None = None, timeout: float = 0.2) -> bool:
        msg = self.message_bus.receive("Verifier", timeout=timeout)
        if not msg:
            return False
        if session_id is not None:
            msg_session = (msg.metadata or {}).get("session_id")
            if msg_session != session_id:
                # Drop stale message belonging to a different session
                return False
        if msg.message_type == MessageType.TASK_RESPONSE:
            try:
                agent_response = self.agent_executor.invoke({"input": str(msg.payload)})
            except Exception as exc:
                verified = f"Verification error: {exc}\n\n{msg.payload}"
            else:
                verified = agent_response.get("output") if isinstance(agent_response, dict) else str(agent_response)
            original_metadata = msg.metadata or {}
            out = Message(
                sender="Verifier",
                recipient="Planner",
                message_type=MessageType.TASK_RESULT,
                payload=verified,
                metadata={
                    "from_task_id": original_metadata.get("task_id"),
                    "session_id": original_metadata.get("session_id", session_id),
                },
            )
            self.message_bus.send(out)
        return True


def create_verifier():
    """Orchestrator mode: return the LangChain-based verifier executor."""
    tools = [
        llm_summarize_tool(
            name="Condense",
            description="Condense and clean up multi-part results into a single coherent answer.",
        )
    ]
    return make_react_agent(
        tools=tools,
        llm=make_llm(temp=0),
        system_prompt=VERIFIER_SYSTEM_PROMPT,
    )


def create_verifier_a2a(message_bus: MessageBus) -> VerifierA2A:
    return VerifierA2A(message_bus)