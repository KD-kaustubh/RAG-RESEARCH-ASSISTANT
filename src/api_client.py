"""HTTP client for the FastAPI backend.

This module only talks to the API. It never loads PDFs, builds embeddings,
touches FAISS or calls Gemini.
"""

import logging
from typing import Any, Dict, Optional

import requests

try:
    from .config import API_BASE_URL, API_TIMEOUT
except ImportError:
    from config import API_BASE_URL, API_TIMEOUT

logger = logging.getLogger(__name__)

HEALTH_TIMEOUT = 5


class APIError(Exception):
    """Carries a message that is safe to show to the user."""


class RagAPIClient:
    def __init__(
        self,
        base_url: str = API_BASE_URL,
        timeout: int = API_TIMEOUT,
        session: Optional[requests.Session] = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.session = session or requests.Session()

    def health(self) -> bool:
        try:
            response = self.session.get(f"{self.base_url}/health", timeout=HEALTH_TIMEOUT)
            return response.status_code == 200
        except requests.RequestException as exc:
            logger.warning("Health check failed for %s: %s", self.base_url, exc)
            return False

    def ask(
        self,
        query: str,
        k: Optional[int] = None,
        session_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {"query": query}
        if k is not None:
            payload["k"] = k
        if session_id:
            payload["session_id"] = session_id

        try:
            response = self.session.post(
                f"{self.base_url}/ask",
                json=payload,
                timeout=self.timeout,
            )
        except requests.Timeout as exc:
            logger.warning("Request to %s timed out: %s", self.base_url, exc)
            raise APIError("The assistant took too long to respond. Please try again.")
        except requests.ConnectionError as exc:
            logger.warning("Could not connect to %s: %s", self.base_url, exc)
            raise APIError(f"Cannot reach the API at {self.base_url}. Is the backend running?")
        except requests.RequestException as exc:
            logger.exception("Request to the API failed")
            raise APIError("The request to the assistant failed.")

        if response.status_code >= 400:
            raise APIError(self._error_message(response))

        return self._parse_answer(response)

    def upload(self, filename: str, content: bytes) -> Dict[str, Any]:
        try:
            response = self.session.post(
                f"{self.base_url}/upload",
                files={"file": (filename, content, "application/pdf")},
                timeout=self.timeout,
            )
        except requests.Timeout as exc:
            logger.warning("Upload to %s timed out: %s", self.base_url, exc)
            raise APIError("Indexing the document took too long. Please try again.")
        except requests.ConnectionError as exc:
            logger.warning("Could not connect to %s: %s", self.base_url, exc)
            raise APIError(f"Cannot reach the API at {self.base_url}. Is the backend running?")
        except requests.RequestException:
            logger.exception("Upload request failed")
            raise APIError("The upload failed.")

        if response.status_code >= 400:
            raise APIError(self._error_message(response))

        try:
            data = response.json()
        except ValueError:
            logger.warning("API returned a non JSON upload response")
            raise APIError("The API returned an unexpected response.")

        if not isinstance(data, dict) or data.get("status") != "ok":
            raise APIError("The API returned an unexpected response.")
        return data

    def _error_message(self, response: requests.Response) -> str:
        detail = None
        try:
            detail = response.json().get("detail")
        except ValueError:
            logger.warning("Non JSON error body from API (status %s)", response.status_code)

        if response.status_code == 422:
            return "The question was rejected by the API. Please rephrase and try again."
        if isinstance(detail, str) and detail:
            return detail
        if response.status_code >= 500:
            return "The assistant is currently unavailable. Please try again shortly."
        return "The API rejected the request."

    def _parse_answer(self, response: requests.Response) -> Dict[str, Any]:
        try:
            data = response.json()
        except ValueError:
            logger.warning("API returned a non JSON success response")
            raise APIError("The API returned an unexpected response.")

        if not isinstance(data, dict) or not isinstance(data.get("answer"), str):
            logger.warning("API response missing the expected answer field")
            raise APIError("The API returned an unexpected response.")

        sources = data.get("sources")
        if not isinstance(sources, list):
            sources = []

        return {
            "answer": data["answer"],
            "sources": sources,
            "session_id": data.get("session_id"),
            "model": data.get("model"),
        }
