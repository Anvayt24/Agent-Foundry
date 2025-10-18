from config import GEMINI_MODEL, RAG_DB_PATH, ensure_google_key
from langchain.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI
from RAG.retriever import get_retriever

ensure_google_key()

# initializing retriver and llm once
retriever = get_retriever(str(RAG_DB_PATH))
llm = ChatGoogleGenerativeAI(model=GEMINI_MODEL, temp=0.1)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff"
)

#Function wrapper
def rag_tool(query: str) -> str:
    return qa_chain.run(query)

try:
    from langchain_core.tools import tool
except Exception:
    from langchain.tools import tool

@tool("rag_answer", return_direct=False)

def rag_answer_tool(query: str) -> str:
    """Retrieve and generate an answer using RAG given a query and context."""
    return rag_tool(query)
