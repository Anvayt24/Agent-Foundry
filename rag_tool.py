import os
from langchain.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI
from retriever import get_retriever
from dotenv import load_dotenv

load_dotenv()

# API key
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise RuntimeError("GEMINI_API_KEY not found. Add it to your .env or OS environment.")
os.environ["GOOGLE_API_KEY"] = api_key

# initializing retriver and llm once
retriever = get_retriever("rag_db")  # persistent vector DB
llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro", temp=0.1)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff"
)

# Function wrapper
def rag_tool(query: str) -> str:
    """Answer a user query using RAG over the local vector store with Gemini."""
    return qa_chain.run(query)


try:
    from langchain_core.tools import tool
except Exception:
    from langchain.tools import tool

@tool("rag_answer", return_direct=False)
def rag_answer_tool(query: str) -> str:
    """Answer a user query using RAG over the local vector store with Gemini."""
    return rag_tool(query)


if __name__ == "__main__":
    test_query = "What is the purpose of this repo?"
    answer = rag_tool(test_query)
    print(f"\nQ: {test_query}\nA: {answer}")
