from central import make_react_agent, llm_summarize_tool

VERIFIER_PROMPT = """
You are the Verifier. You receive multiple worker outputs.
1) Check for consistency, missing pieces, and obvious errors.
2) If needed, synthesize a concise final answer (cite snippets inline if present).
Return a clean final answer.

Worker Outputs:
{input}
{agent_scratchpad}
"""

def create_verifier():
    tools = [llm_summarize_tool(name="Condense", description="Condense multi-part results into a clean answer.")]
    return make_react_agent(prompt_template=VERIFIER_PROMPT, tools=tools, temp=0)