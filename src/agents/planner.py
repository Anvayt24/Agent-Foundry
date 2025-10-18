from langchain.tools import Tool
from core.central import make_llm, make_react_agent
import json
import time
from core.messaging import MessageBus, Message, MessageType
from memory.memory_manager import memory_manager

PLANNER_SYSTEM_PROMPT = """
You are the Planner Agent. Your job is to create a plan with subtasks for the user's request.

You MUST end with "Final Answer:" followed by ONLY JSON in this format:
{{"subtasks": ["step 1", "step 2", "step 3"]}}

Available tools: search_memories, add_memory, plan_task

Workflow:
1. If the request references past information, use search_memories
2. Use plan_task if you need help generating subtasks
3. ALWAYS end with "Final Answer:" and the JSON

IMPORTANT TOOL RULE:
If the user explicitly requests a specific tool or capability (e.g., "use RAG tool", "call RAG_Search"), ensure at least one subtask tells the worker to invoke that exact tool while addressing the request.

Use this format:
Thought: [your reasoning]
Action: [tool name]
Action Input: [tool parameters]
Observation: [tool result]
... (repeat as needed)
Final Answer: {{"subtasks": ["step 1", "step 2"]}}

NEVER use "skip" or any other action. When you're ready to provide the plan, use "Final Answer:" with JSON.
"""

class PlannerA2A:
    def __init__(self, message_bus: MessageBus):
        self.message_bus = message_bus
        self.message_bus.register_agent("Planner")
        # Initialize result tracking
        self.expected_results: int = 0
        self.results: list[str] = []

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
            
    def process_user_request(self, user_request: str):
        """Process user request by creating and dispatching subtasks."""
        subtasks = self.create_subtasks(user_request)
        self.expected_results = len(subtasks)
        self.results = []
        
        for i, task in enumerate(subtasks, 1):
            msg = Message(
                sender="Planner",
                recipient="Worker",
                message_type=MessageType.TASK_REQUEST,
                payload=task,
                metadata={
                    "task_id": f"task_{i}", 
                    "total": len(subtasks), 
                    "original_request": user_request
                },
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
        llm=make_llm(temp=0),
        system_prompt=PLANNER_SYSTEM_PROMPT,
    )

def create_planner_a2a(message_bus: MessageBus) -> PlannerA2A:
    return PlannerA2A(message_bus)