from langchain_chroma import Chroma
from RAG.vector_store import build_vector_store, get_embeddings

def get_retriever(persist_directory="rag_db"):
    embeddings = get_embeddings()
    vectordb = Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings
    )
    return vectordb.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 2, "fetch_k": 10, "lambda_mult": 0.5}
    )

if __name__ == "__main__":
    
    print("Building vector store...")
    build_vector_store(".")
    print("Vector store built successfully!")
    
    print("\nTesting retriever...")
    retriever = get_retriever()
    results = retriever.invoke("What is the purpose of this repo?")

    seen_contents = set()
    for r in results:
        content = r.page_content.strip()
        if content in seen_contents:
            continue
        seen_contents.add(content)
        print(content, "\n---")