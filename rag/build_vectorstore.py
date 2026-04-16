"""
Run this ONCE to build and save the FAISS vectorstore.
Command: python rag/build_vectorstore.py
"""

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


def build_vectorstore() -> None:
    """Load maintenance guidelines, chunk them, and persist a FAISS index."""
    loader = TextLoader("rag/guidelines/maintenance_manual.txt")
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    chunks = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local("rag/vectorstore")

    print(
        f"Vectorstore built with {len(chunks)} chunks. "
        "Saved to rag/vectorstore/."
    )


if __name__ == "__main__":
    build_vectorstore()
