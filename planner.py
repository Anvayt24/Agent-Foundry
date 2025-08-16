from langchain.tools import Tool
from central import make_llm, make_react_agent
import json

PLANNER_SYSTEM_PROMPT = """
You are the Planner Agent. Your job is to break the user's objective into a minimal ordered list of atomic subtasks.

IMPORTANT: You must respond with ONLY valid JSON in this exact format:
{"subtasks": ["step 1", "step 2", "step 3"]}

Rules:
1. Output ONLY the JSON, no other text
2. Use the exact format above with double quotes
3. Each subtask should be a simple, actionable step
4. If a step needs retrieval from knowledge base, include "RAG" in the task description
5. If a step is pure reasoning/summarization, include "SUMMARIZE" in the task description

Always follow the ReAct format:
Thought: reasoning about how to break down the task
Action: plan_task
Action Input: the planning prompt
Observation: the planning result
Final Answer: the final JSON plan
"""

def create_planner():
    # Create a planning tool that handles JSON validation
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
        
        # Try to clean up the response if it's not valid JSON
        if not output.startswith('{'):
            start = output.find('{')
            end = output.rfind('}') + 1
            if start != -1 and end > start:
                output = output[start:end]
        
        # Validate JSON format
        try:
            json.loads(output)
            return output
        except json.JSONDecodeError:
            # If JSON is invalid, create a simple fallback
            return json.dumps({
                "subtasks": [
                    f"Use RAG to search for information about: {input_text}",
                    "SUMMARIZE the findings into a clear explanation"
                ]
            })
    
    # Create planning tool
    planning_tool = Tool(
        name="plan_task",
        func=plan_task,
        description="Break down user objectives into ordered subtasks and return as JSON"
    )
    
    tools = [planning_tool]
    return make_react_agent(
        tools=tools,
        llm=make_llm(temp=0),
        system_prompt=PLANNER_SYSTEM_PROMPT
    )