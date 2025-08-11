from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

def load_and_chunk_doc (path: str, chunk_size = 500, chunk_overlap = 50):
    loader = DirectoryLoader(path, glob="*.txt", loader_cls= TextLoader)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size = chunk_size,
        chunk_overlap = chunk_overlap
    )
    return splitter.split_documents(docs)
     

