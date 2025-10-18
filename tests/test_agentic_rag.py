from config import GEMINI_MODEL, ensure_google_key
from langchain.tools import Tool
from langchain.agents import create_react_agent, AgentExecutor
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from RAG.rag_tool import rag_tool

ensure_google_key()

rag_search_tool = Tool(
    name="RAG Search",
    func=lambda query: rag_tool(query),
    description="Use this tool to search the document knowledge base and get relevant context for a user query."
)

llm = ChatGoogleGenerativeAI(model=GEMINI_MODEL, temp=0)

# ReAct agent
tools = [rag_search_tool]
prompt = PromptTemplate.from_template("""
You are an intelligent AI assistant with access to tools.
Your job is to answer user questions accurately.

You have access to the following tools:
{tools}

Tool names: {tool_names}

Use the following format:
Question: The input question you must answer
Thought: Reason about what to do next
Action: The tool to use (must be exactly the name of the tool)
Action Input: The input to the tool
Observation: The tool's result
... (this Thought/Action/Observation loop can repeat)
Final Answer: The final answer to the original question

Begin!

Question: {input}
{agent_scratchpad}
""")
agent = create_react_agent(llm=llm, tools=tools, prompt=prompt)

# Wrap in AgentExecutor
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True) #why

if __name__ == "__main__":
    print("\nAgentic RAG (Gemini) ready! Ask anything about your docs.\n")
    while True:
        query = input("You: ")
        if query.lower() in ["exit", "quit"]:
            break
        response = agent_executor.invoke({"input": query})
        print("Agent:", response["output"])
 