from langchain.tools import Tool
from langchain.prompts import PromptTemplate
from langchain.agents import initialize_agent, AgentType
from langchain_google_genai import ChatGoogleGenerativeAI
import os 
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise RuntimeError("GEMINI_API_KEY not found. Add it to your .env file.")
os.environ["GOOGLE_API_KEY"] = api_key

def make_llm(temp: float = 0):
    return ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=temp)

def make_react_agent(tools: list, llm, system_prompt: str, temp: float = 0):
    """
    Creates a ReAct-style AgentExecutor with the given tools, LLM, and system prompt.
    Always returns an AgentExecutor so .invoke() works.
    """
    prompt = PromptTemplate(
        template=system_prompt,
        input_variables=["input", "agent_scratchpad"]
    )

    return initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        handle_parsing_errors=True
    )

def llm_summarize_tool(name="Summarize", description="Summarize text succinctly."):
    llm = make_llm(0)
    return Tool(
        name=name,
        description=description,
        func=lambda text: llm.invoke(f"Summarize clearly and briefly:\n\n{text}").content,
    )

def run_agent(agent_executor, input_text: str) -> str:
    try:
        return agent_executor.invoke({"input": input_text})["output"]
    except Exception as e:
        return f"Error running agent: {str(e)}"
