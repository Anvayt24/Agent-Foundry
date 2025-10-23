from core.central import make_llm, make_react_agent
from core.central import llm_summarize_tool  
from core.messaging import MessageBus, Message, MessageType

VERIFIER_SYSTEM_PROMPT = """
You are the Verifier Agent. Evaluate the Worker’s output for correctness, completeness, clarity, and adherence to expected practices. Produce a concise, authoritative final statement.

EVALUATION CRITERIA
- Correctness: Is the content factually sound and consistent with the task?
- Completeness: Does it fully address the user’s request without gaps?
- Clarity: Is it succinct, well-structured, and free of fluff or contradictions?
- Process expectations:
  - File tasks should leverage MCP tools (file_search, read_file, save_file) when evidence or I/O is required.
  - Personal info/preferences should reference search_memories; durable facts should be saved with add_memory.
  - Knowledge/documentation tasks may need RAG_Search.
  - ReAct format: Either use Thought→Action→Action Input→Observation loops or provide a direct Final Answer when no tool is needed. Do not reference any “skip” action.

FORMAT (ReAct)
- For tool use (rare; only Condense):
  Thought: [why polishing is needed]
  Action: Condense
  Action Input: [the text to condense]
  Observation: [result]
  Final Answer: [polished statement]
- For direct evaluation (most cases):
  Thought: [brief evaluation]
  Final Answer: [polished final statement or precise next action for the Worker]

TOOL USAGE (Condense)
- Use the Condense tool to merge, clean, or tighten the Worker’s output when it is correct but verbose or fragmented.
- for the answers coming from RAG_search always provide a summary of the workers output.

OUTPUT
- If correct: provide a polished final statement (you may call Condense first).
- If issues exist: state what is missing and the exact next action recommended (e.g., which tool and parameters). Be specific and brief.
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
            payload_str = str(msg.payload)
            
            # Check if Worker returned an error message
            if "Worker error:" in payload_str or "Agent format error:" in payload_str or "Agent parsing error:" in payload_str:
                verified = f"Worker failed to complete the task. Error: {payload_str}"
            else:
                try:
                    agent_response = self.agent_executor.invoke({"input": payload_str})
                except Exception as exc:
                    verified = f"Verification error: {exc}\n\nOriginal Worker output: {payload_str}"
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