from typing import Optional
from fastapi import FastAPI
from pydantic import BaseModel

try:
    from .config import (
        PDF_PATH,
        TOP_K,
        CHUNK_SIZE,
        CHUNK_OVERLAP,
        FAISS_INDEX_PATH,
    )
    from .llm import get_llm
    from .rag_core import ask_question as run_rag_query, sources_from_docs
    from .vector_store import get_vectorstore
except ImportError:
    from config import (
        PDF_PATH,
        TOP_K,
        CHUNK_SIZE,
        CHUNK_OVERLAP,
        FAISS_INDEX_PATH,
    )
    from llm import get_llm
    from rag_core import ask_question as run_rag_query, sources_from_docs
    from vector_store import get_vectorstore

app = FastAPI()

vectorstore = get_vectorstore(
    pdf_path=PDF_PATH,
    index_path=FAISS_INDEX_PATH,
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
)
llm = get_llm()


class QueryRequest(BaseModel):
    query: str
    k: Optional[int] = None


@app.post("/ask")
def ask_question(request: QueryRequest):
    k = request.k or TOP_K
    answer, docs = run_rag_query(request.query, vectorstore, llm, k=k)

    return {
        "answer": answer,
        "sources": sources_from_docs(docs),
    }
