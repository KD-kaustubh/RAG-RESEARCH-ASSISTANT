import pytest
from fastapi.testclient import TestClient

import main
from conftest import FakeLLM, FakeVectorStore


@pytest.fixture
def resources():
    """Pre-load fake resources so startup never calls Gemini."""
    store = FakeVectorStore()
    llm = FakeLLM()
    main._resources.clear()
    main._resources["vectorstore"] = store
    main._resources["llm"] = llm
    main.session_store = main.SessionStore()
    yield store, llm
    main._resources.clear()


@pytest.fixture
def client(resources):
    with TestClient(main.app) as test_client:
        yield test_client


def test_health_returns_ok():
    with TestClient(main.app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_ask_returns_answer_and_sources(client):
    response = client.post("/ask", json={"query": "What is the Transformer?"})

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "A fake answer."
    assert body["sources"][0]["page"] == 1
    assert body["sources"][0]["excerpt"]
    assert body["session_id"] is None


def test_ask_respects_requested_k(client, resources):
    store, _ = resources
    client.post("/ask", json={"query": "What is this?", "k": 1})

    assert store.last_k == 1


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"query": ""},
        {"query": "hi", "k": 0},
        {"query": "hi", "k": 99},
        {"query": "hi", "session_id": ""},
    ],
)
def test_invalid_requests_are_rejected(client, payload):
    assert client.post("/ask", json=payload).status_code == 422


def test_llm_failure_returns_bad_gateway(client, resources):
    _, llm = resources
    llm.error = RuntimeError("gemini is down")

    response = client.post("/ask", json={"query": "What is this?"})

    assert response.status_code == 502
    assert "gemini is down" not in response.text


def test_missing_api_key_returns_service_unavailable(monkeypatch):
    main._resources.clear()

    def missing_key():
        raise RuntimeError("Missing GOOGLE_API_KEY. Set it in .env or your environment.")

    monkeypatch.setattr(main, "load_resources", missing_key)
    with TestClient(main.app) as client:
        response = client.post("/ask", json={"query": "What is this?"})

    assert response.status_code == 503
    assert "GOOGLE_API_KEY" in response.json()["detail"]


def test_missing_document_returns_service_unavailable(monkeypatch):
    main._resources.clear()

    def missing_pdf():
        raise FileNotFoundError("PDF not found: paper.pdf")

    monkeypatch.setattr(main, "load_resources", missing_pdf)
    with TestClient(main.app) as client:
        response = client.post("/ask", json={"query": "What is this?"})

    assert response.status_code == 503
    assert "PDF not found" in response.json()["detail"]


def test_session_history_is_sent_to_the_model(client, resources):
    _, llm = resources
    client.post("/ask", json={"query": "What is the Transformer?", "session_id": "s1"})
    client.post("/ask", json={"query": "What are its parts?", "session_id": "s1"})

    assert "What is the Transformer?" in llm.prompts[1]
    assert "Previous conversation" in llm.prompts[1]


def test_first_request_has_no_history(client, resources):
    _, llm = resources
    client.post("/ask", json={"query": "What is the Transformer?", "session_id": "s1"})

    assert "Previous conversation" not in llm.prompts[0]


def test_sessions_are_isolated(client, resources):
    _, llm = resources
    client.post("/ask", json={"query": "question for A", "session_id": "A"})
    client.post("/ask", json={"query": "question for B", "session_id": "B"})

    assert "question for A" not in llm.prompts[1]


def test_requests_without_session_stay_stateless(client):
    client.post("/ask", json={"query": "first"})
    client.post("/ask", json={"query": "second"})

    assert main.session_store.session_count() == 0


def test_session_response_echoes_session_id(client):
    response = client.post("/ask", json={"query": "hello", "session_id": "abc"})

    assert response.json()["session_id"] == "abc"
