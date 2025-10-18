from core.central import make_llm, make_react_agent
from core.central import llm_summarize_tool  
from core.messaging import MessageBus, Message, MessageType

VERIFIER_SYSTEM_PROMPT = """
You are the Verifier Agent.
Your job is to check the combined worker outputs for correctness, completeness, and clarity.
If errors or gaps are found, fix them in the final answer.

Always follow the ReAct format:

Thought: reasoning
Action: tool_name (if needed, else skip)
Action Input: the input
Observation: tool output
Final Answer: the verified and corrected result

When no tool is needed, do NOT output an Action step. Go directly from Thought to Final Answer.
If the worker's response is already correct, repeat or lightly improve that content in your Final Answer—never respond with a bare acknowledgement such as "Okay" or "Noted".
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