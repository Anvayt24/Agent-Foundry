import os
from langchain.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI
from retriever import get_retriever

# Expects GOOGLE_API_KEY in environment

def rag_tool(query: str, persist_directory: str = "rag_db", model: str = "gemini-1.5-flash", temperature: float = 0.1) -> str:
    """
    Run a Retrieval-Augmented Generation query using Gemini models via LangChain.
    Set GOOGLE_API_KEY in environment before use.
    """
    retriever = get_retriever(persist_directory)
    llm = ChatGoogleGenerativeAI(model=model, temperature=temperature)

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff"
    )
    return qa_chain.run(query)

# Optional: Tool wrapper for agent frameworks
try:
    from langchain_core.tools import tool
except Exception:
    from langchain.tools import tool

@tool("rag_answer", return_direct=False)
def rag_answer_tool(query: str) -> str:
    """Answer a user query using RAG over the local vector store with Gemini."""
    return rag_tool(query)

if __name__ == "__main__":
    # Ensure GOOGLE_API_KEY is set in your environment before running
    test_query = "What is the purpose of this ?"
    answer = rag_tool(test_query)
    print(f"\nQ: {test_query}\nA: {answer}")
