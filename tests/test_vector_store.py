import os
from pathlib import Path

import pytest
from langchain_core.documents import Document

import vector_store
from vector_store import _index_exists, build_vectorstore, get_vectorstore, split_documents

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


def page(number, text):
    return Document(page_content=text, metadata={"page": number, "source": "test.pdf"})


def test_sentence_spanning_two_pages_stays_in_one_chunk():
    pages = [
        page(0, "Each layer has two sub-layers. The first is a multi-head self-attention"),
        page(1, "mechanism, and the second is a feed-forward network."),
    ]

    chunks = split_documents(pages, chunk_size=500, chunk_overlap=50)

    assert any(
        "multi-head self-attention" in c.page_content and "feed-forward network" in c.page_content
        for c in chunks
    )


def test_production_phrase_split_across_pages_is_recoverable():
    """The exact failure seen in production: the phrase was cut at the page break."""
    pages = [
        page(0, "The first is a multi-head self-attention mechanism, and the second is a simple, position-"),
        page(1, "wise fully connected feed-forward network. We employ a residual connection."),
    ]

    chunks = split_documents(pages, chunk_size=500, chunk_overlap=50)

    combined = [c.page_content for c in chunks]
    assert any("position-" in c and "wise fully connected feed-forward network" in c for c in combined)


def test_chunk_spanning_pages_records_every_page():
    pages = [page(3, "first half of the sentence"), page(4, "second half of the sentence")]

    chunks = split_documents(pages, chunk_size=500, chunk_overlap=0)

    spanning = [c for c in chunks if "first half" in c.page_content and "second half" in c.page_content]
    assert spanning
    assert spanning[0].metadata["pages"] == [3, 4]


def test_chunk_page_metadata_points_at_first_contributing_page():
    pages = [page(3, "first half of the sentence"), page(4, "second half of the sentence")]

    chunk = split_documents(pages, chunk_size=500, chunk_overlap=0)[0]

    assert chunk.metadata["page"] == 3


def test_single_page_chunk_keeps_its_own_page():
    pages = [page(0, "alpha " * 60), page(1, "beta " * 60)]

    chunks = split_documents(pages, chunk_size=100, chunk_overlap=0)
    beta_chunks = [c for c in chunks if "beta" in c.page_content and "alpha" not in c.page_content]

    assert beta_chunks
    assert all(c.metadata["page"] == 1 for c in beta_chunks)
    assert all(c.metadata["pages"] == [1] for c in beta_chunks)


def test_split_documents_preserves_source_metadata():
    chunks = split_documents([page(0, "some text")], chunk_size=100, chunk_overlap=0)

    assert chunks[0].metadata["source"] == "test.pdf"


def test_split_documents_handles_empty_input():
    assert split_documents([], chunk_size=100, chunk_overlap=0) == []


def real_pdf_chunks():
    pdf = Path(__file__).resolve().parents[1] / "paper.pdf"
    if not pdf.exists():
        pytest.skip("paper.pdf not available")

    from loader import load_pdf

    return split_documents(load_pdf(str(pdf)), chunk_size=500, chunk_overlap=50)


def test_real_pdf_produces_chunks_that_span_pages():
    """Needs no API key: chunking is checked directly, nothing is embedded."""
    assert any(len(c.metadata.get("pages", [])) > 1 for c in real_pdf_chunks())


def test_encoder_sublayer_chunk_continues_past_the_page_break():
    """The production bug: this chunk stopped at the end of its page, mid-sentence."""
    marker = "the second is a simple, position-"
    chunks = [c for c in real_pdf_chunks() if marker in c.page_content]

    assert chunks
    for chunk in chunks:
        # Under the old per-page splitter this chunk came from one page and ended here.
        assert len(chunk.metadata["pages"]) > 1
        after_marker = chunk.page_content.split(marker, 1)[1]
        assert "Figure 1: The Transformer" in after_marker


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
