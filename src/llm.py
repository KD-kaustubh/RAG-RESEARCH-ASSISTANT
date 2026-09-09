import logging
from typing import Optional

from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI

try:
    from .config import (
        GEMINI_CHAT_MODEL,
        GEMINI_EMBED_MODEL,
        GROQ_API_KEY,
        GROQ_CHAT_MODEL,
        require_api_key,
    )
except ImportError:
    from config import (
        GEMINI_CHAT_MODEL,
        GEMINI_EMBED_MODEL,
        GROQ_API_KEY,
        GROQ_CHAT_MODEL,
        require_api_key,
    )

logger = logging.getLogger(__name__)


def get_embeddings(model_name: Optional[str] = None) -> GoogleGenerativeAIEmbeddings:
    require_api_key()
    return GoogleGenerativeAIEmbeddings(model=model_name or GEMINI_EMBED_MODEL)


def get_llm(model_name: Optional[str] = None, temperature: float = 0.2) -> ChatGoogleGenerativeAI:
    require_api_key()
    return ChatGoogleGenerativeAI(model=model_name or GEMINI_CHAT_MODEL, temperature=temperature)


def get_groq_llm(model_name: Optional[str] = None, temperature: float = 0.2):
    if not GROQ_API_KEY:
        raise RuntimeError("Missing GROQ_API_KEY. Set it in .env or your environment.")

    from langchain_groq import ChatGroq

    return ChatGroq(model=model_name or GROQ_CHAT_MODEL, temperature=temperature)


def model_name(llm) -> str:
    """Best-effort identifier of the model behind a LangChain chat object."""
    for attribute in ("model", "model_name"):
        value = getattr(llm, attribute, None)
        if isinstance(value, str) and value:
            return value.replace("models/", "")
    return type(llm).__name__


class FallbackLLM:
    """Answers with the primary model, and switches to the backup when it fails.

    Gemini's free tier allows only a small number of requests per day. Without a
    backup the assistant stops answering entirely once that cap is reached.
    """

    def __init__(self, primary, backup) -> None:
        self.primary = primary
        self.backup = backup
        # Best effort: with concurrent requests this reflects the most recent call.
        self.last_provider: Optional[str] = None
        self.last_model: Optional[str] = None

    def invoke(self, prompt):
        try:
            response = self.primary.invoke(prompt)
        except Exception as exc:
            logger.warning("Primary model failed (%s); using the backup model", type(exc).__name__)
            response = self.backup.invoke(prompt)
            self.last_provider = "backup"
            self.last_model = model_name(self.backup)
            return response

        self.last_provider = "primary"
        self.last_model = model_name(self.primary)
        return response


def describe_model(llm) -> str:
    """Which model produced the most recent answer."""
    if isinstance(llm, FallbackLLM):
        return llm.last_model or model_name(llm.primary)
    return model_name(llm)


def get_chat_model(temperature: float = 0.2):
    """Gemini for answers, falling back to Groq when a GROQ_API_KEY is configured."""
    primary = get_llm(temperature=temperature)
    if not GROQ_API_KEY:
        return primary

    try:
        backup = get_groq_llm(temperature=temperature)
    except Exception:
        logger.exception("Groq backup could not be created; continuing with Gemini only")
        return primary

    logger.info("Groq backup enabled for answer generation")
    return FallbackLLM(primary, backup)
