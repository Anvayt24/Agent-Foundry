from langchain.tools import Tool
from langchain.prompts import PromptTemplate
from langchain.agents import create_react_agent, AgentExecutor
from langchain_google_genai import ChatGoogleGenerativeAI
import os 
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise RuntimeError("GEMINI_API_KEY not found. Add it to your .env file.")
os.environ["GOOGLE_API_KEY"] = api_key

def make_llm(temp: float = 0):
    return ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=temp)

def make_react_agent(prompt_template: str, tools: list, temp: float = 0, verbose: bool = True) -> AgentExecutor:
    llm = make_llm(temp)
    
    # Add missing required variables to the prompt template
    if "{tools}" not in prompt_template:
        prompt_template += "\n\nAvailable tools:\n{tools}\n\nTool names: {tool_names}"
    
    prompt = PromptTemplate.from_template(prompt_template)
    agent = create_react_agent(llm=llm, tools=tools, prompt=prompt)
    return AgentExecutor(agent=agent, tools=tools, verbose=verbose, handle_parsing_errors=True)

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
