from langchain.tools import Tool
from core.central import make_llm, make_react_agent, make_planner_llm
import json
import time
from core.messaging import MessageBus, Message, MessageType
from memory.memory_manager import memory_manager

PLANNER_SYSTEM_PROMPT = """
You are the Planner Agent. Create an executable plan (ordered subtasks) for the user's request.

OBJECTIVE
- Produce a plan the Worker can execute immediately.
- Subtasks must be atomic, unambiguous, sequential, and tool-aware (name the tool the Worker should use, when relevant).
- If no decomposition is needed, return a single subtask mirroring the request.

ALLOWED TOOLS: search_memories, add_memory, plan_task
- search_memories: When the request references prior interactions, user preferences, IDs, or saved facts.
- add_memory: Save newly surfaced preferences/constraints critical for future steps.
- plan_task: Use only to draft subtasks; you MUST still return the final JSON.

OUTPUT REQUIREMENTS
- You MUST end with exactly: Final Answer: followed by ONLY compact JSON in this format:
  {{"subtasks": ["step 1", "step 2", "step 3"]}}
- No extra text, no code fences, no commentary, no trailing punctuation after the JSON.

USER-REQUESTED TOOLS
- If the user explicitly asks to use a tool (e.g., "RAG_Search", "file_search"), include at least one subtask that instructs the Worker to call that exact tool with appropriate parameters.

DEPENDENCY MANAGEMENT
- When later work depends on data produced earlier (e.g., read a file to extract a value, then store that value), keep them in a SINGLE subtask so the Worker can perform both steps within one ReAct chain.
- Example: "Read src/RAG/vector_store.py to identify the embedding model, then call add_memory to store 'Embedding model in src/RAG/vector_store.py: <MODEL_NAME>'."

THINKING/ACTING FORMAT (ReAct)
Thought: [your reasoning]
Action: [tool name]
Action Input: [valid JSON for that tool]
Observation: [result]
... (as needed)
Final Answer: {{"subtasks": ["step 1", "step 2"]}}

CONSTRAINTS
- Do NOT use any fake actions (e.g., "skip").
- Provide the plan only via Final Answer JSON.
"""

class PlannerA2A:
    def __init__(self, message_bus: MessageBus):
        self.message_bus = message_bus
        self.message_bus.register_agent("Planner")
        # Initialize result tracking
        self.expected_results: int = 0
        self.results: list[str] = []
        self.active_session_id: str | None = None

    def create_subtasks(self, user_request: str) -> list:
        """Create subtasks using the agent."""
        agent = create_planner()
        try:
            response = agent.invoke({"input": user_request})
            plan = response.get("output") if isinstance(response, dict) else response
            plan_data = json.loads(plan) if isinstance(plan, str) else plan
            subtasks = plan_data.get("subtasks", [user_request]) if isinstance(plan_data, dict) else [user_request]
            return subtasks
        except Exception as e:
            print(f"Error in create_subtasks: {e}")
            return [user_request]
            
    def process_user_request(self, user_request: str, session_id: str | None = None):
        """Process user request by creating and dispatching subtasks."""
        subtasks = self.create_subtasks(user_request)
        self.expected_results = len(subtasks)
        self.results = []
        self.active_session_id = session_id
        
        for i, task in enumerate(subtasks, 1):
            msg = Message(
                sender="Planner",
                recipient="Worker",
                message_type=MessageType.TASK_REQUEST,
                payload=task,
                metadata={
                    "task_id": f"task_{i}", 
                    "total": len(subtasks), 
                    "original_request": user_request,
                    "session_id": session_id,
                },
            )
            self.message_bus.send(msg)

    def collect_results(self, session_id: str | None = None, max_wait_seconds: float = 1.0) -> str:
        """Poll the message bus for TASK_RESULT messages until all expected
        results are gathered or the timeout expires, then combine them.
        """
        target_session = session_id if session_id is not None else self.active_session_id
        deadline = time.time() + max_wait_seconds
        while time.time() < deadline and len(self.results) < self.expected_results:
            msg = self.message_bus.receive("Planner", timeout=0.05)
            if not msg:
                continue
            if msg.message_type == MessageType.TASK_RESULT:
                msg_session = (msg.metadata or {}).get("session_id")
                if target_session and msg_session != target_session:
                    continue
                self.results.append(str(msg.payload))

        if not self.results:
            return ""

        if len(self.results) == 1:
            return self.results[0]

        return "\n\n".join(self.results)


def create_planner():
    """Orchestrator mode: return a LangChain ReAct planner agent (backward compatible)."""
    def plan_task(input_text: str) -> str:
        llm = make_planner_llm(temp=0)
        prompt = f"""
        Break down this objective into subtasks: {input_text}
        
        Return ONLY valid JSON in this format:
        {{"subtasks": ["step 1", "step 2", "step 3"]}}
        """
        response = llm.invoke(prompt)
        output = response.content if hasattr(response, "content") else response
        output = output.strip()
        if not output.startswith('{'):
            start = output.find('{')
            end = output.rfind('}') + 1
            if start != -1 and end > start:
                output = output[start:end]
        try:
            json.loads(output)
            return output
        except json.JSONDecodeError:
            return json.dumps({
                "subtasks": [
                    f"{input_text}"
                ]
            })

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

    planning_tool = Tool(
        name="plan_task",
        func=plan_task,
        description="Break down user objectives into ordered subtasks and return as JSON",
    )
    tools = [planning_tool]
    tools = tools + memory_tools
    return make_react_agent(
        tools=tools,
        llm=make_planner_llm(temp=0),
        system_prompt=PLANNER_SYSTEM_PROMPT,
    )

def create_planner_a2a(message_bus: MessageBus) -> PlannerA2A:
    return PlannerA2A(message_bus)