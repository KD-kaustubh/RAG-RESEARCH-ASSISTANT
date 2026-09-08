import os

import pytest
from langchain_core.documents import Document

import vector_store
from vector_store import _index_exists, build_vectorstore, get_vectorstore

LONG_PAGE = "Attention is all you need. " * 60


@pytest.fixture
def sample_pages():
    return [
        Document(page_content=LONG_PAGE, metadata={"page": 0, "source": "test.pdf"}),
        Document(page_content="The decoder stack also has six layers. " * 20, metadata={"page": 1, "source": "test.pdf"}),
    ]


@pytest.fixture
def fake_pdf(tmp_path):
    """A placeholder file; parsing is stubbed out by the patched_loader fixture."""
    path = tmp_path / "test.pdf"
    path.write_bytes(b"%PDF-1.4")
    return str(path)


@pytest.fixture
def patched_loader(monkeypatch, sample_pages, fake_embeddings):
    """Run the real chunking/FAISS code without a PDF file or Gemini."""
    monkeypatch.setattr(vector_store, "load_pdf", lambda path: sample_pages)
    monkeypatch.setattr(vector_store, "get_embeddings", lambda *args, **kwargs: fake_embeddings)
    return sample_pages


def test_index_exists_false_for_empty_dir(tmp_path):
    assert _index_exists(str(tmp_path)) is False


def test_index_exists_true_when_both_files_present(tmp_path):
    (tmp_path / "index.faiss").write_text("x")
    (tmp_path / "index.pkl").write_text("x")

    assert _index_exists(str(tmp_path)) is True


def test_index_exists_false_when_pkl_missing(tmp_path):
    (tmp_path / "index.faiss").write_text("x")

    assert _index_exists(str(tmp_path)) is False


def test_build_splits_pages_into_smaller_chunks(patched_loader, fake_embeddings, fake_pdf):
    store = build_vectorstore(fake_pdf, fake_embeddings, chunk_size=200, chunk_overlap=20)

    chunks = list(store.docstore._dict.values())
    assert len(chunks) > len(patched_loader)
    assert all(len(chunk.page_content) <= 200 for chunk in chunks)


def test_chunks_keep_page_metadata(patched_loader, fake_embeddings, fake_pdf):
    store = build_vectorstore(fake_pdf, fake_embeddings, chunk_size=200, chunk_overlap=20)

    pages = {chunk.metadata.get("page") for chunk in store.docstore._dict.values()}
    assert pages == {0, 1}


def test_build_rejects_missing_pdf(fake_embeddings):
    with pytest.raises(FileNotFoundError):
        build_vectorstore("does_not_exist.pdf", fake_embeddings)


def test_similarity_search_returns_documents(patched_loader, fake_embeddings, fake_pdf):
    store = build_vectorstore(fake_pdf, fake_embeddings, chunk_size=200, chunk_overlap=20)

    results = store.similarity_search("decoder stack layers", k=2)
    assert len(results) == 2
    assert all(isinstance(doc, Document) for doc in results)


def test_get_vectorstore_persists_index(patched_loader, fake_pdf, tmp_path):
    index_path = str(tmp_path / "index")

    get_vectorstore(pdf_path=fake_pdf, index_path=index_path)

    assert os.path.exists(os.path.join(index_path, "index.faiss"))
    assert os.path.exists(os.path.join(index_path, "index.pkl"))


def test_get_vectorstore_reloads_saved_index(monkeypatch, patched_loader, fake_pdf, tmp_path):
    index_path = str(tmp_path / "index")
    get_vectorstore(pdf_path=fake_pdf, index_path=index_path)

    def fail_if_rebuilt(*args, **kwargs):
        raise AssertionError("index should have been loaded from disk")

    monkeypatch.setattr(vector_store, "load_pdf", fail_if_rebuilt)
    store = get_vectorstore(pdf_path=fake_pdf, index_path=index_path)

    assert store.similarity_search("decoder", k=1)


def test_get_vectorstore_rebuild_flag_ignores_saved_index(patched_loader, fake_pdf, tmp_path):
    index_path = str(tmp_path / "index")
    get_vectorstore(pdf_path=fake_pdf, index_path=index_path)

    calls = []
    original = vector_store.load_pdf

    def counting_loader(path):
        calls.append(path)
        return original(path)

    vector_store.load_pdf = counting_loader
    try:
        get_vectorstore(pdf_path=fake_pdf, index_path=index_path, rebuild=True)
    finally:
        vector_store.load_pdf = original

    assert len(calls) == 1
