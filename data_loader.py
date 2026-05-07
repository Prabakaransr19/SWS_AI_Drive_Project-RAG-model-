from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
import os

DOCS_PATH = "Docs/"
CHROMA_PATH = "chroma_db"

def load_and_store():
    # 1. Load all PDFs
    all_docs = []
    for file in os.listdir(DOCS_PATH):
        if file.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(DOCS_PATH, file))
            all_docs.extend(loader.load())
            print(f"Loaded: {file}")

    # 2. Chunk
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = splitter.split_documents(all_docs)
    print(f"\nTotal chunks: {len(chunks)}")

    # 3. Embed + Store in Chroma
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_PATH
    )
    print(f"\nStored in Chroma at '{CHROMA_PATH}/'")

if __name__ == "__main__":
    load_and_store()