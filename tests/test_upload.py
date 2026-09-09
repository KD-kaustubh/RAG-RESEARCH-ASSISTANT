"""Upload tests. Indexing is stubbed, so no API key, network or Gemini call is needed."""

import pytest
from fastapi.testclient import TestClient
from langchain_core.documents import Document

import main
from conftest import FakeLLM, FakeVectorStore


class FakeIndex:
    ntotal = 7


class IndexedStore(FakeVectorStore):
    """Stands in for a freshly built FAISS store for the uploaded document."""

    def __init__(self, docs=None):
        super().__init__(docs)
        self.index = FakeIndex()


def minimal_pdf() -> bytes:
    return b"%PDF-1.4\n1 0 obj\n<<>>\nendobj\ntrailer\n<<>>\n%%EOF\n"


@pytest.fixture
def client(tmp_path, monkeypatch):
    """Fresh app state, an isolated upload directory, and stubbed indexing."""
    monkeypatch.setattr(main, "UPLOAD_DIR", str(tmp_path / "uploads"))
    monkeypatch.setattr(main, "session_store", main.SessionStore())

    built = {}

    def fake_get_vectorstore(pdf_path, **kwargs):
        built["pdf_path"] = pdf_path
        built["rebuild"] = kwargs.get("rebuild")
        return IndexedStore(
            [Document(page_content="Uploaded paper content.", metadata={"page": 2})]
        )

    monkeypatch.setattr(main, "get_vectorstore", fake_get_vectorstore)
    monkeypatch.setattr(main, "get_chat_model", lambda: FakeLLM(answer="From the uploaded paper."))

    main._resources.clear()
    main._resources["vectorstore"] = FakeVectorStore()
    main._resources["llm"] = FakeLLM()

    with TestClient(main.app) as test_client:
        test_client.built = built
        yield test_client

    main._resources.clear()


def upload(client, name="paper.pdf", content=None, content_type="application/pdf"):
    body = minimal_pdf() if content is None else content
    return client.post("/upload", files={"file": (name, body, content_type)})


def test_valid_pdf_upload_succeeds(client):
    response = upload(client)

    assert response.status_code == 200
    body = response.json()
    assert body == {"status": "ok", "filename": "paper.pdf", "chunks": 7}


def test_upload_rebuilds_the_index_from_the_stored_file(client):
    upload(client)

    assert client.built["rebuild"] is True
    assert client.built["pdf_path"].endswith("active.pdf")


def test_uploaded_document_becomes_the_active_one(client):
    before = client.post("/ask", json={"query": "What is this about?"}).json()
    assert "Uploaded paper content." not in before["sources"][0]["excerpt"]

    upload(client)
    after = client.post("/ask", json={"query": "What is this about?"})

    assert after.status_code == 200
    # Retrieval now serves the uploaded document instead of the default paper.
    assert after.json()["sources"][0]["excerpt"] == "Uploaded paper content."


def test_sources_from_the_uploaded_document_keep_page_numbers(client):
    upload(client)

    sources = client.post("/ask", json={"query": "What is this about?"}).json()["sources"]

    assert sources[0]["page"] == 3  # zero-indexed page 2 is reported as page 3
    assert sources[0]["excerpt"] == "Uploaded paper content."


def test_upload_clears_earlier_conversations(client):
    client.post("/ask", json={"query": "about the old paper", "session_id": "s1"})
    assert main.session_store.get_history("s1")

    upload(client)

    assert main.session_store.get_history("s1") == []


def test_sessions_still_work_after_upload(client):
    upload(client)
    client.post("/ask", json={"query": "first", "session_id": "s2"})
    client.post("/ask", json={"query": "second", "session_id": "s2"})

    assert len(main.session_store.get_history("s2")) == 4


def test_non_pdf_is_rejected(client):
    response = upload(client, name="notes.txt", content=b"hello", content_type="text/plain")

    assert response.status_code == 400
    assert "PDF" in response.json()["detail"]


def test_empty_file_is_rejected(client):
    response = upload(client, content=b"")

    assert response.status_code == 400
    assert "empty" in response.json()["detail"].lower()


def test_file_without_pdf_header_is_rejected(client):
    response = upload(client, content=b"this is not really a pdf")

    assert response.status_code == 400
    assert "not a valid PDF" in response.json()["detail"]


def test_oversized_upload_is_rejected(client, monkeypatch):
    monkeypatch.setattr(main, "MAX_UPLOAD_MB", 1)

    response = upload(client, content=b"%PDF-1.4" + b"0" * (1024 * 1024 + 1))

    assert response.status_code == 413
    assert "larger than" in response.json()["detail"]


def test_unreadable_pdf_reports_a_client_error(client, monkeypatch):
    from pypdf.errors import PdfReadError

    def broken(*args, **kwargs):
        raise PdfReadError("damaged file")

    monkeypatch.setattr(main, "get_vectorstore", broken)
    response = upload(client)

    assert response.status_code == 400
    assert response.json()["detail"] == "The PDF could not be read."
    assert "damaged file" not in response.text


def test_missing_api_key_during_upload_reports_service_unavailable(client, monkeypatch):
    def missing_key(*args, **kwargs):
        raise RuntimeError("Missing GOOGLE_API_KEY. Set it in .env or your environment.")

    monkeypatch.setattr(main, "get_vectorstore", missing_key)
    response = upload(client)

    assert response.status_code == 503
    assert "GOOGLE_API_KEY" in response.json()["detail"]


def test_indexing_failure_does_not_leak_internals(client, monkeypatch):
    def boom(*args, **kwargs):
        raise ValueError("connection to internal host 10.0.0.5 failed")

    monkeypatch.setattr(main, "get_vectorstore", boom)
    response = upload(client)

    assert response.status_code == 502
    assert response.json()["detail"] == "Could not index the uploaded document."
    assert "10.0.0.5" not in response.text


def test_client_filename_is_never_used_as_a_path(client):
    response = upload(client, name="../../evil.pdf")

    assert response.status_code == 200
    assert response.json()["filename"] == "evil.pdf"
    assert client.built["pdf_path"].endswith("active.pdf")
