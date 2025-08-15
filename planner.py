from central import make_react_agent , Tool

PLANNER_PROMPT = """
You are the Planner. Your job is to break the user's objective into a minimal ordered list of atomic subtasks.

IMPORTANT: You must respond with ONLY valid JSON in this exact format:
{{"subtasks": ["step 1", "step 2", "step 3"]}}

Rules:
1. Output ONLY the JSON, no other text
2. Use the exact format above with double quotes
3. Each subtask should be a simple, actionable step
4. If a step needs retrieval from knowledge base, include "RAG" in the task description
5. If a step is pure reasoning/summarization, include "SUMMARIZE" in the task description

Question: {input}
{agent_scratchpad}
"""

# Planner uses no external tools—pure reasoning
def create_planner():
    return make_react_agent(prompt_template=PLANNER_PROMPT, tools=[])