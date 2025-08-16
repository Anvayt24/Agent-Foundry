from langchain.tools import Tool
from central import make_llm, make_react_agent
from rag_tool import rag_tool

WORKER_SYSTEM_PROMPT = """
You are a Worker Agent.
You execute the subtask assigned to you using the tools provided.
Always follow the ReAct format:

Thought: reasoning
Action: tool_name
Action Input: the input
Observation: tool output
Final Answer: the final completed subtask output
"""

def create_worker():
    # Create RAG tool wrapper
    rag_tool_wrapper = Tool(
        name="RAG_Search",
        func=rag_tool,
        description="Search the document knowledge base for relevant information"
    )
    
    tools = [rag_tool_wrapper]
    return make_react_agent(
        tools=tools,
        llm=make_llm(temp=0),
        system_prompt=WORKER_SYSTEM_PROMPT
    )
