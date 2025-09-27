from langchain_chroma import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from RAG.load_docs import load_and_chunk_doc
from pathlib import Path
from typing import Optional

_EMBEDDINGS: Optional[SentenceTransformerEmbeddings] = None


def get_embeddings(model_name: str = "all-MiniLM-L6-v2") -> SentenceTransformerEmbeddings:
    global _EMBEDDINGS
    if _EMBEDDINGS is None:
        _EMBEDDINGS = SentenceTransformerEmbeddings(model_name=model_name)
    return _EMBEDDINGS


def build_vector_store(data_path: str, persist_directory="rag_db"):
    persist_path = Path(persist_directory)
    embeddings = get_embeddings()

    if persist_path.exists() and any(persist_path.iterdir()):
        return Chroma(
            persist_directory=str(persist_path),
            embedding_function=embeddings
        )

    docs = load_and_chunk_doc(data_path)
    vectordb = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=str(persist_path)
    )
    vectordb.persist()
    return vectordb
