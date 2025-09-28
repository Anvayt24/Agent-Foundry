from langchain.tools import Tool
from central import make_llm, make_react_agent
import json
import time
from messaging import MessageBus, Message, MessageType

PLANNER_SYSTEM_PROMPT = """
You are the Planner Agent. Your job is to break the user's objective into a minimal ordered list of atomic subtasks.

IMPORTANT: You must respond with ONLY valid JSON in this exact format:
{"subtasks": ["step 1", "step 2", "step 3"]}

Rules:
1. Output ONLY the JSON, no other text.
2. Use the exact format above with double quotes.
3. Each subtask should be a simple, actionable step.
4. Include "RAG" in a subtask ONLY when information is needed from the project knowledge base (docs, code, repo-specific context). Do NOT include RAG for general knowledge questions.
5. Include "MCP" in a subtask ONLY for repository/file operations (e.g., search files, read_file, save_file).
6. If the user request is a general conceptual question that can be answered directly, return a single subtask with the original question (no RAG, no MCP).

Always follow the ReAct format:
Thought: reasoning about how to break down the task
Action: plan_task
Action Input: the planning prompt
Observation: the planning result
Final Answer: the final JSON plan
"""

class PlannerA2A:
    def __init__(self, message_bus: MessageBus):
        self.message_bus = message_bus
        self.message_bus.register_agent("Planner")
        # Initialize result tracking
        self.expected_results: int = 0
        self.results: list[str] = []

    def _plan_task(self, input_text: str) -> str:
        llm = make_llm(temp=0)
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
            # Fallback: avoid forcing RAG; let the Worker decide tool usage.
            return json.dumps({
                "subtasks": [
                    f"{input_text}"
                ]
            })

    def create_subtasks(self, user_request: str) -> list:
        plan_json = self._plan_task(user_request)
        try:
            plan = json.loads(plan_json)
            return plan.get("subtasks", [user_request])
        except Exception:
            return [user_request]

    def process_user_request(self, user_request: str):
        subtasks = self.create_subtasks(user_request)
        # Reset tracking counters for this user request
        self.expected_results = len(subtasks)
        self.results = []
        for i, task in enumerate(subtasks, 1):
            msg = Message(
                sender="Planner",
                recipient="Worker",
                message_type=MessageType.TASK_REQUEST,
                payload=task,
                metadata={"task_id": f"task_{i}", "total": len(subtasks), "original_request": user_request},
            )
            self.message_bus.send(msg)

    def collect_results(self, max_wait_seconds: float = 1.0) -> str:
        """Poll the message bus for TASK_RESULT messages until all expected
        results are gathered or the timeout expires, then combine them.
        """
        deadline = time.time() + max_wait_seconds
        while time.time() < deadline and len(self.results) < self.expected_results:
            msg = self.message_bus.receive("Planner", timeout=0.05)
            if not msg:
                continue
            if msg.message_type == MessageType.TASK_RESULT:
                self.results.append(str(msg.payload))

        if not self.results:
            return ""

        if len(self.results) == 1:
            return self.results[0]

        return "\n\n".join(self.results)


def create_planner():
    """Orchestrator mode: return a LangChain ReAct planner agent (backward compatible)."""
    def plan_task(input_text: str) -> str:
        llm = make_llm(temp=0)
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

    planning_tool = Tool(
        name="plan_task",
        func=plan_task,
        description="Break down user objectives into ordered subtasks and return as JSON",
    )
    tools = [planning_tool]
    return make_react_agent(
        tools=tools,
        llm=make_llm(temp=0),
        system_prompt=PLANNER_SYSTEM_PROMPT,
    )

def create_planner_a2a(message_bus: MessageBus) -> PlannerA2A:
    return PlannerA2A(message_bus)