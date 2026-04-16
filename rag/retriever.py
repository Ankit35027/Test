import os

from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


_vectorstore = None


load_dotenv()


def _build_embeddings() -> HuggingFaceEmbeddings:
    token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN")
    model_kwargs = {"token": token} if token else {}
    return HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs=model_kwargs,
    )


def get_retriever():
    """Load the FAISS store once and reuse it across calls."""
    global _vectorstore

    if _vectorstore is None:
        embeddings = _build_embeddings()
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
