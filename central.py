from langchain.tools import Tool
from langchain.agents import AgentExecutor, create_react_agent
from langchain.prompts import PromptTemplate
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


def make_react_agent(tools, llm, system_prompt, temp: float = 0):
    template = (
        f"{system_prompt}\n\n"
        "You can use the following tools:\n{tools}\n\n"
        "Tool names: {tool_names}\n\n"
        "Question: {input}\n"
        "{agent_scratchpad}"
    )
    prompt = PromptTemplate(
        template=template,
        input_variables=["input", "agent_scratchpad", "tools", "tool_names"],
    )
    agent = create_react_agent(llm=llm, tools=tools, prompt=prompt)
    return AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=3,
        early_stopping_method="generate",
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
        return agent_executor.invoke({
            "input": input_text
        })["output"]
    except Exception as e:
        return f"Error running agent: {str(e)}"
