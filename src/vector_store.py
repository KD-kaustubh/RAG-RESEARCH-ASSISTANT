import os
from typing import Optional
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

try:
    from .config import CHUNK_SIZE, CHUNK_OVERLAP, FAISS_INDEX_PATH, GEMINI_EMBED_MODEL
    from .loader import load_pdf
    from .llm import get_embeddings
except ImportError:
    from config import CHUNK_SIZE, CHUNK_OVERLAP, FAISS_INDEX_PATH, GEMINI_EMBED_MODEL
    from loader import load_pdf
    from llm import get_embeddings


def _index_exists(index_path: str) -> bool:
    return os.path.exists(os.path.join(index_path, "index.faiss")) and os.path.exists(
        os.path.join(index_path, "index.pkl")
    )


def build_vectorstore(
    pdf_path: str,
    embeddings,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> FAISS:
    if not os.path.exists(pdf_path):
        raise SystemExit(f"PDF not found: {pdf_path}")

    documents = load_pdf(pdf_path)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunks = text_splitter.split_documents(documents)
    return FAISS.from_documents(chunks, embeddings)


def get_vectorstore(
    pdf_path: str,
    index_path: str = FAISS_INDEX_PATH,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
    embedding_model: Optional[str] = None,
    rebuild: bool = False,
) -> FAISS:
    embeddings = get_embeddings(embedding_model or GEMINI_EMBED_MODEL)

    if not rebuild and _index_exists(index_path):
        return FAISS.load_local(
            index_path,
            embeddings,
            allow_dangerous_deserialization=True,
        )

    vectorstore = build_vectorstore(
        pdf_path=pdf_path,
        embeddings=embeddings,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    vectorstore.save_local(index_path)
    return vectorstore
