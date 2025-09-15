from central import make_llm, make_react_agent
from central import llm_summarize_tool  

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

def create_verifier():
    tools = [
        llm_summarize_tool(
            name="Condense",
            description="Condense and clean up multi-part results into a single coherent answer."
        )
    ]
    return make_react_agent(
        tools=tools,
        llm=make_llm(temp=0),
        system_prompt=VERIFIER_SYSTEM_PROMPT
    )