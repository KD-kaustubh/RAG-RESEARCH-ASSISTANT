import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from pypdf.errors import PdfReadError

try:
    from .config import (
        ALLOWED_ORIGINS,
        CHUNK_OVERLAP,
        CHUNK_SIZE,
        FAISS_INDEX_PATH,
        MAX_HISTORY_MESSAGES,
        MAX_UPLOAD_MB,
        PDF_PATH,
        TOP_K,
        UPLOAD_DIR,
    )
    from .llm import describe_model, get_chat_model
    from .rag_core import ask_question as run_rag_query, format_history, sources_from_docs
    from .session_store import SessionStore
    from .vector_store import get_vectorstore
except ImportError:
    from config import (
        ALLOWED_ORIGINS,
        CHUNK_OVERLAP,
        CHUNK_SIZE,
        FAISS_INDEX_PATH,
        MAX_HISTORY_MESSAGES,
        MAX_UPLOAD_MB,
        PDF_PATH,
        TOP_K,
        UPLOAD_DIR,
    )
    from llm import describe_model, get_chat_model
    from rag_core import ask_question as run_rag_query, format_history, sources_from_docs
    from session_store import SessionStore
    from vector_store import get_vectorstore

logger = logging.getLogger(__name__)

_resources: dict = {}
session_store = SessionStore()


def load_resources() -> dict:
    """Build the vector store and LLM once, then reuse them for later requests."""
    if not _resources:
        _resources["vectorstore"] = get_vectorstore(
            pdf_path=PDF_PATH,
            index_path=FAISS_INDEX_PATH,
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
        )
        _resources["llm"] = get_chat_model()
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
    session_id: Optional[str] = Field(None, min_length=1, max_length=64)


class Source(BaseModel):
    page: Optional[int] = None
    excerpt: str


class AskResponse(BaseModel):
    answer: str
    sources: List[Source]
    session_id: Optional[str] = None
    model: Optional[str] = None


class UploadResponse(BaseModel):
    status: str
    filename: str
    chunks: int


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)):
    """Replace the active document with an uploaded PDF and rebuild the index."""
    filename = Path(file.filename or "").name
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="The uploaded file is empty.")
    if len(content) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(status_code=413, detail=f"The file is larger than {MAX_UPLOAD_MB} MB.")
    if not content.startswith(b"%PDF"):
        raise HTTPException(status_code=400, detail="The uploaded file is not a valid PDF.")

    # A fixed server-side name: the client filename is only ever shown back, never used as a path.
    upload_dir = Path(UPLOAD_DIR)
    upload_dir.mkdir(parents=True, exist_ok=True)
    stored_pdf = upload_dir / "active.pdf"
    stored_pdf.write_bytes(content)

    try:
        vectorstore = get_vectorstore(
            pdf_path=str(stored_pdf),
            index_path=FAISS_INDEX_PATH,
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            rebuild=True,
        )
        llm = _resources.get("llm") or get_chat_model()
    except PdfReadError:
        logger.exception("Uploaded PDF could not be parsed")
        raise HTTPException(status_code=400, detail="The PDF could not be read.")
    except RuntimeError as exc:
        logger.exception("Configuration error while indexing the upload")
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception:
        logger.exception("Failed to index the uploaded document")
        raise HTTPException(status_code=502, detail="Could not index the uploaded document.")

    _resources["vectorstore"] = vectorstore
    _resources["llm"] = llm
    # A new paper must not inherit conversations about the previous one.
    session_store.clear_all()

    return {
        "status": "ok",
        "filename": filename,
        "chunks": getattr(vectorstore.index, "ntotal", 0),
    }


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

    history_text = ""
    if request.session_id:
        history = session_store.get_history(request.session_id)
        history_text = format_history(history, limit=MAX_HISTORY_MESSAGES)

    try:
        answer, docs = run_rag_query(
            request.query,
            resources["vectorstore"],
            resources["llm"],
            k=k,
            history_text=history_text,
        )
    except Exception:
        logger.exception("Failed to answer question")
        raise HTTPException(status_code=502, detail="Failed to generate an answer.")

    if request.session_id:
        session_store.add_turn(request.session_id, request.query, answer)

    return {
        "answer": answer,
        "sources": sources_from_docs(docs),
        "session_id": request.session_id,
        "model": describe_model(resources["llm"]),
    }
