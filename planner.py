from central import make_react_agent , Tool

PLANNER_PROMPT = """
You are the Planner. Break the user's objective into a minimal ordered list of atomic subtasks.
Only output strict JSON with this schema:
{"subtasks": ["step 1", "step 2", "..."]}

If a step needs retrieval from the knowledge base, include the word RAG in it.
If a step is pure reasoning/summarization, include the word SUMMARIZE.

Question: {input}
{agent_scratchpad}
"""

# Planner uses no external tools—pure reasoning
def create_planner():
    return make_react_agent(prompt_template=PLANNER_PROMPT, tools=[])