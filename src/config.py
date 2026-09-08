import os
from dotenv import load_dotenv

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

GEMINI_EMBED_MODEL = os.getenv("GEMINI_EMBED_MODEL", "gemini-embedding-001")
GEMINI_CHAT_MODEL = os.getenv("GEMINI_CHAT_MODEL", "gemini-2.5-flash")

PDF_PATH = os.getenv("PDF_PATH", os.path.join("data", "paper.pdf"))
QUERY_DEFAULT = os.getenv("QUERY", "What is the main idea of the paper?")

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))
TOP_K = int(os.getenv("TOP_K", "3"))

FAISS_INDEX_PATH = os.getenv("FAISS_INDEX_PATH", "faiss_index")

_DEFAULT_ALLOWED_ORIGINS = "http://localhost:8501,http://127.0.0.1:8501"
ALLOWED_ORIGINS = [
    origin.strip()
    for origin in os.getenv("ALLOWED_ORIGINS", _DEFAULT_ALLOWED_ORIGINS).split(",")
    if origin.strip()
]


def require_api_key() -> None:
    if not GOOGLE_API_KEY:
        raise RuntimeError("Missing GOOGLE_API_KEY. Set it in .env or your environment.")
