from config import (
    PLANNER_MODEL, WORKER_MODEL, VERIFIER_MODEL,
    MODEL_BACKEND, OLLAMA_BASE_URL,
    GEMINI_MODEL, USE_GEMINI_FALLBACK,
    ensure_google_key
)
from langchain.tools import Tool
from langchain.agents import AgentExecutor, create_react_agent
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

ensure_google_key()

def make_llm(temp: float = 0, model_name: str | None = None, agent_type: str | None = None):
    """
    Create an LLM instance (Ollama or Gemini).
    
    Args:
        temp: Temperature setting (0-1)
        model_name: Specific model to use (overrides agent_type)
        agent_type: Agent type ('planner', 'worker', 'verifier') for automatic model selection
    
    Returns:
        ChatOllama or ChatGoogleGenerativeAI instance
    """
    # Determine which model to use
    if model_name:
        model = model_name
    elif agent_type == "planner":
        model = PLANNER_MODEL
    elif agent_type == "worker":
        model = WORKER_MODEL
    elif agent_type == "verifier":
        model = VERIFIER_MODEL
    else:
        model = PLANNER_MODEL  # Default fallback
    
    try:
        if MODEL_BACKEND == "ollama":
            from langchain_ollama import ChatOllama
            return ChatOllama(
                model=model,
                temperature=temp,
                base_url=OLLAMA_BASE_URL
            )
        
        elif MODEL_BACKEND == "gemini":
            ensure_google_key()
            return ChatGoogleGenerativeAI(model=model, temperature=temp)
        
        else:
            raise ValueError(f"Unsupported MODEL_BACKEND: {MODEL_BACKEND}")

    except Exception as e:
        if USE_GEMINI_FALLBACK:
            print(f"Warning: Failed to load {MODEL_BACKEND}:{model}, falling back to Gemini. Error: {e}")
            ensure_google_key()
            return ChatGoogleGenerativeAI(model=GEMINI_MODEL, temperature=temp)
        else:
            raise

# Convenience functions for agent-specific LLMs
def make_planner_llm(temp: float = 0):
    return make_llm(temp=temp, agent_type="planner")

def make_worker_llm(temp: float = 0):
    return make_llm(temp=temp, agent_type="worker")

def make_verifier_llm(temp: float = 0):
    return make_llm(temp=temp, agent_type="verifier")


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
