import os
from langchain.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI
from RAG.retriever import get_retriever
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise RuntimeError("GEMINI_API_KEY not found. Add it to your .env or OS environment.")
os.environ["GOOGLE_API_KEY"] = api_key

# initializing retriver and llm once
retriever = get_retriever("rag_db")  #persistent vector 
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temp=0.1)

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
