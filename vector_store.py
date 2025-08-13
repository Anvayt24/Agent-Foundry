from langchain_chroma import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from load_docs import load_and_chunk_doc

def build_vector_store(data_path: str, persist_directory="rag_db"):
    docs = load_and_chunk_doc(data_path)
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    vectordb = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=persist_directory
    )
    vectordb.persist()
    return vectordb


