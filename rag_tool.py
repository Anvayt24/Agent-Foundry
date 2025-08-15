import os
from langchain.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI
from retriever import get_retriever
from dotenv import load_dotenv

load_dotenv()

def rag_tool(query: str, persist_directory: str = "rag_db", model: str = "gemini-2.5-pro", temp: float = 0.1) -> str:

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not found. Add it to your .env or OS environment.")

    # Set the environment variable that langchain-google-genai expects
    os.environ["GOOGLE_API_KEY"] = api_key

    retriever = get_retriever(persist_directory)
    llm = ChatGoogleGenerativeAI(
        model=model, 
        temp=temp
    )

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
    test_query = "What is the purpose of this ?"
    answer = rag_tool(test_query)
    print(f"\nQ: {test_query}\nA: {answer}")
