"""A small fixed evaluation set for the Transformer paper.

This checks that the chunks the retriever can return actually contain the
information each question needs, and that page metadata stays right. It is a
regression aid, not a production feature, and it never calls Gemini: chunking is
deterministic, so no API key, network or FAISS index is required.
"""

from pathlib import Path

import pytest

from vector_store import split_documents

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# question, phrases one chunk must contain together, pages that chunk may come from
EVAL_SET = [
    (
        "What is the main idea of the paper?",
        ["relying entirely on an attention mechanism"],
        {0, 1},
    ),
    (
        "What are the two main components of the Transformer?",
        ["the encoder and decoder"],
        {1, 2},
    ),
    (
        "What are the two sublayers in each encoder layer?",
        ["multi-head self-attention mechanism", "position-wise fully connected feed-forward network"],
        {1, 2},
    ),
    (
        "What are the two sublayers in each decoder layer?",
        ["inserts a third sub-layer"],
        {2},
    ),
    (
        "Why does the Transformer avoid recurrence?",
        ["precludes parallelization"],
        {0, 1},
    ),
    (
        "What is the role of positional encoding?",
        ["positional encodings"],
        {4, 5, 6},
    ),
    (
        "How many identical layers are in the encoder and decoder stacks?",
        ["stack of N = 6 identical layers"],
        {1, 2},
    ),
    (
        "What is the purpose of multi-head attention?",
        ["jointly attend to information from different representation"],
        {3},
    ),
]


@pytest.fixture(scope="module")
def chunks():
    pdf = Path(__file__).resolve().parents[1] / "paper.pdf"
    if not pdf.exists():
        pytest.skip("paper.pdf not available")

    from loader import load_pdf

    return split_documents(load_pdf(str(pdf)), chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)


@pytest.mark.parametrize("question,phrases,pages", EVAL_SET, ids=[q for q, _, _ in EVAL_SET])
def test_a_single_chunk_answers_the_question(chunks, question, phrases, pages):
    matching = [c for c in chunks if all(phrase in c.page_content for phrase in phrases)]

    assert matching, f"no chunk contains all of {phrases}"
    assert any(set(c.metadata.get("pages", [])) & pages for c in matching), (
        f"expected a chunk from pages {sorted(pages)}"
    )


def test_every_chunk_carries_page_metadata(chunks):
    assert all(isinstance(c.metadata.get("page"), int) for c in chunks)
    assert all(c.metadata.get("pages") for c in chunks)


def test_chunks_stay_within_the_configured_size(chunks):
    assert all(len(c.page_content) <= CHUNK_SIZE for c in chunks)
