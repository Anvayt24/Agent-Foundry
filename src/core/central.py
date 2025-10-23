from config import GEMINI_MODEL, ensure_google_key
from langchain.tools import Tool
from langchain.agents import AgentExecutor, create_react_agent
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

ensure_google_key()

def make_llm(temp: float = 0):
    return ChatGoogleGenerativeAI(model=GEMINI_MODEL, temperature=temp)


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
        max_iterations=10,
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
