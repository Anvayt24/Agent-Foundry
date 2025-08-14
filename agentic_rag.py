import os
from langchain.tools import Tool
from langchain.prompts import PromptTemplate
from langchain.agents import create_react_agent, AgentExecutor
from langchain_google_genai import ChatGoogleGenerativeAI
from rag_tool import rag_tool
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set up API key for langchain-google-genai
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise RuntimeError("GEMINI_API_KEY not found. Add it to your .env file.")
os.environ["GOOGLE_API_KEY"] = api_key

rag_search_tool = Tool(
    name="RAG Search",
    func=lambda query: rag_tool(query),
    description="Use this tool to search the document knowledge base and get relevant context for a user query."
)

prompt = PromptTemplate.from_template("""
You are an intelligent AI assistant with access to tools.
Your job is to answer user questions accurately.

You can use the following tools:
{tools}

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

llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)

# 4️⃣ Build the ReAct agent
tools = [rag_search_tool]
agent = create_react_agent(llm=llm, tools=tools, prompt=prompt)

# 5️⃣ Wrap in AgentExecutor
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

if __name__ == "__main__":
    print("\nAgentic RAG (Gemini) ready! Ask anything about your docs.\n")
    while True:
        query = input("You: ")
        if query.lower() in ["exit", "quit"]:
            break
        response = agent_executor.invoke({"input": query})
        print("Agent:", response["output"])
 