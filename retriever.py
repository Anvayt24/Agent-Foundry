from langchain_community.vectorstores import Chroma
from langchain.embeddings import SentenceTransformerEmbeddings

def get_retriever(persist_directory="rag_db"):
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    vectordb = Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings
    )
    return vectordb.as_retriever(search_kwargs={"k": 3})    # returns 3 most relevant chunks

if __name__ == "__main__":
    retriever = get_retriever()
    results = retriever.get_relevant_documents("What is the purpose of this repo?")
    for r in results:
        print(r.page_content[:200], "\n---")