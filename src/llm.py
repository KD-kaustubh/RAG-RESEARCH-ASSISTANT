from typing import Optional
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI

try:
    from .config import GEMINI_EMBED_MODEL, GEMINI_CHAT_MODEL, require_api_key
except ImportError:
    from config import GEMINI_EMBED_MODEL, GEMINI_CHAT_MODEL, require_api_key


def get_embeddings(model_name: Optional[str] = None) -> GoogleGenerativeAIEmbeddings:
    require_api_key()
    return GoogleGenerativeAIEmbeddings(model=model_name or GEMINI_EMBED_MODEL)


def get_llm(model_name: Optional[str] = None, temperature: float = 0.2) -> ChatGoogleGenerativeAI:
    require_api_key()
    return ChatGoogleGenerativeAI(model=model_name or GEMINI_CHAT_MODEL, temperature=temperature)
