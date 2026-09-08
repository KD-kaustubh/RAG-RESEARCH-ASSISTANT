import pytest
import requests

from api_client import APIError, RagAPIClient


class FakeResponse:
    def __init__(self, status_code=200, payload=None, text_body=None):
        self.status_code = status_code
        self._payload = payload
        self._text_body = text_body

    def json(self):
        if self._text_body is not None:
            raise ValueError("not json")
        return self._payload


class FakeSession:
    """Records calls and returns queued responses or raises a queued error."""

    def __init__(self, response=None, error=None):
        self.response = response
        self.error = error
        self.calls = []

    def _handle(self, method, url, **kwargs):
        self.calls.append({"method": method, "url": url, **kwargs})
        if self.error:
            raise self.error
        return self.response

    def get(self, url, **kwargs):
        return self._handle("GET", url, **kwargs)

    def post(self, url, **kwargs):
        return self._handle("POST", url, **kwargs)


def make_client(response=None, error=None):
    session = FakeSession(response=response, error=error)
    return RagAPIClient(base_url="http://testserver", session=session), session


def test_base_url_trailing_slash_is_removed():
    client, _ = make_client()
    assert RagAPIClient(base_url="http://testserver/", session=client.session).base_url == "http://testserver"


def test_health_true_on_200():
    client, session = make_client(FakeResponse(200, {"status": "ok"}))

    assert client.health() is True
    assert session.calls[0]["url"] == "http://testserver/health"


def test_health_false_on_error_status():
    client, _ = make_client(FakeResponse(503, {"detail": "down"}))

    assert client.health() is False


def test_health_false_when_backend_unreachable():
    client, _ = make_client(error=requests.ConnectionError("refused"))

    assert client.health() is False


def test_ask_returns_answer_and_sources():
    payload = {
        "answer": "The Transformer uses attention.",
        "sources": [{"page": 2, "excerpt": "text"}],
        "session_id": "s1",
    }
    client, session = make_client(FakeResponse(200, payload))

    result = client.ask("What is the Transformer?", k=2, session_id="s1")

    assert result["answer"] == "The Transformer uses attention."
    assert result["sources"][0]["page"] == 2
    assert result["session_id"] == "s1"
    assert session.calls[0]["url"] == "http://testserver/ask"
    assert session.calls[0]["json"] == {
        "query": "What is the Transformer?",
        "k": 2,
        "session_id": "s1",
    }


def test_ask_omits_optional_fields_when_not_given():
    client, session = make_client(FakeResponse(200, {"answer": "hi", "sources": []}))

    client.ask("hello")

    assert session.calls[0]["json"] == {"query": "hello"}


def test_ask_raises_friendly_error_when_backend_down():
    client, _ = make_client(error=requests.ConnectionError("refused"))

    with pytest.raises(APIError) as exc:
        client.ask("hello")

    assert "Cannot reach the API" in str(exc.value)


def test_ask_raises_friendly_error_on_timeout():
    client, _ = make_client(error=requests.Timeout("too slow"))

    with pytest.raises(APIError) as exc:
        client.ask("hello")

    assert "too long" in str(exc.value)


def test_ask_reports_server_detail():
    client, _ = make_client(FakeResponse(503, {"detail": "RAG service is not available."}))

    with pytest.raises(APIError) as exc:
        client.ask("hello")

    assert "RAG service is not available." in str(exc.value)


def test_ask_gives_generic_message_for_validation_error():
    client, _ = make_client(FakeResponse(422, {"detail": [{"loc": ["body", "query"]}]}))

    with pytest.raises(APIError) as exc:
        client.ask("")

    assert "rephrase" in str(exc.value)


def test_ask_handles_server_error_without_json_body():
    client, _ = make_client(FakeResponse(500, text_body="<html>boom</html>"))

    with pytest.raises(APIError) as exc:
        client.ask("hello")

    assert "unavailable" in str(exc.value)
    assert "html" not in str(exc.value)


def test_ask_rejects_non_json_success_response():
    client, _ = make_client(FakeResponse(200, text_body="not json"))

    with pytest.raises(APIError) as exc:
        client.ask("hello")

    assert "unexpected response" in str(exc.value)


def test_ask_rejects_response_without_answer():
    client, _ = make_client(FakeResponse(200, {"sources": []}))

    with pytest.raises(APIError):
        client.ask("hello")


def test_ask_tolerates_missing_sources():
    client, _ = make_client(FakeResponse(200, {"answer": "hi"}))

    assert client.ask("hello")["sources"] == []
