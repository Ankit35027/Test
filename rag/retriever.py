from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


_vectorstore = None


def get_retriever():
    """Load the FAISS store once and reuse it across calls."""
    global _vectorstore

    if _vectorstore is None:
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        _vectorstore = FAISS.load_local(
            "rag/vectorstore",
            embeddings,
            allow_dangerous_deserialization=True,
        )

    return _vectorstore.as_retriever(search_kwargs={"k": 4})


def retrieve_guidelines(query: str) -> str:
    """Return the most relevant maintenance guidance as one string."""
    retriever = get_retriever()
    docs = retriever.invoke(query)
    return "\n\n".join(doc.page_content for doc in docs)
