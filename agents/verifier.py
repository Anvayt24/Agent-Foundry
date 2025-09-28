from central import make_llm, make_react_agent
from central import llm_summarize_tool  
from messaging import MessageBus, Message, MessageType
import time

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
"""

class VerifierA2A:
    def __init__(self, message_bus: MessageBus):
        self.message_bus = message_bus
        self.message_bus.register_agent("Verifier")

    def _verify(self, text: str) -> str:
        llm = make_llm(0)
        try:
            resp = llm.invoke(f"Clean up and verify this result for correctness and clarity. If unclear, improve it.\n\n{text}")
            return resp.content if hasattr(resp, "content") else str(resp)
        except Exception as e:
            return f"Verification error: {e}\n\n{text}"

    def process_once(self, timeout: float = 0.2) -> bool:
        msg = self.message_bus.receive("Verifier", timeout=timeout)
        if not msg:
            return False
        if msg.message_type == MessageType.TASK_RESPONSE:
            verified = self._verify(str(msg.payload))
            out = Message(
                sender="Verifier",
                recipient="Planner",
                message_type=MessageType.TASK_RESULT,
                payload=verified,
                metadata={"from_task_id": (msg.metadata or {}).get("task_id")},
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