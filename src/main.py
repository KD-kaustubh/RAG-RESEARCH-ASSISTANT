import logging
from contextlib import asynccontextmanager
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

try:
    from .config import (
        ALLOWED_ORIGINS,
        CHUNK_OVERLAP,
        CHUNK_SIZE,
        FAISS_INDEX_PATH,
        PDF_PATH,
        TOP_K,
    )
    from .llm import get_llm
    from .rag_core import ask_question as run_rag_query, sources_from_docs
    from .vector_store import get_vectorstore
except ImportError:
    from config import (
        ALLOWED_ORIGINS,
        CHUNK_OVERLAP,
        CHUNK_SIZE,
        FAISS_INDEX_PATH,
        PDF_PATH,
        TOP_K,
    )
    from llm import get_llm
    from rag_core import ask_question as run_rag_query, sources_from_docs
    from vector_store import get_vectorstore

logger = logging.getLogger(__name__)

_resources: dict = {}


def load_resources() -> dict:
    """Build the vector store and LLM once, then reuse them for later requests."""
    if not _resources:
        _resources["vectorstore"] = get_vectorstore(
            pdf_path=PDF_PATH,
            index_path=FAISS_INDEX_PATH,
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
        )
        _resources["llm"] = get_llm()
    return _resources


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        load_resources()
    except Exception:
        # Startup stays up so /health works; /ask reports the problem per request.
        logger.exception("RAG resources could not be initialized at startup")
    yield
    _resources.clear()


app = FastAPI(title="RAG Research Assistant API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    k: Optional[int] = Field(None, ge=1, le=20)


class Source(BaseModel):
    page: Optional[int] = None
    excerpt: str


class AskResponse(BaseModel):
    answer: str
    sources: List[Source]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
def ask_question(request: QueryRequest):
    try:
        resources = load_resources()
    except RuntimeError as exc:
        logger.exception("Configuration error while preparing RAG resources")
        raise HTTPException(status_code=503, detail=str(exc))
    except FileNotFoundError as exc:
        logger.exception("Source document missing")
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception:
        logger.exception("Failed to initialize RAG resources")
        raise HTTPException(status_code=503, detail="RAG service is not available.")

    k = request.k or TOP_K

    try:
        answer, docs = run_rag_query(
            request.query,
            resources["vectorstore"],
            resources["llm"],
            k=k,
        )
    except Exception:
        logger.exception("Failed to answer question")
        raise HTTPException(status_code=502, detail="Failed to generate an answer.")

    return {"answer": answer, "sources": sources_from_docs(docs)}
