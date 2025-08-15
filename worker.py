# agents/worker.py
from langchain.tools import Tool
from .common import make_react_agent, llm_summarize_tool
from rag_tool import rag_tool

WORKER_PROMPT = """
You are the Worker. Execute the single subtask provided using available tools.
Use tools when appropriate. Return ONLY the final result for this subtask.

Subtask: {input}
{agent_scratchpad}
"""

def create_worker():
    tools = [
        Tool(
            name="RAG Search",
            func=lambda q: rag_tool(q),
            description="Query the document knowledge base and return relevant context/answers."
        ),
        llm_summarize_tool(),
    ]
    return make_react_agent(prompt_template=WORKER_PROMPT, tools=tools, temp=0)
