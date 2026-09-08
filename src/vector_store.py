import os
from typing import List, Optional
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

try:
    from .config import CHUNK_SIZE, CHUNK_OVERLAP, FAISS_INDEX_PATH, GEMINI_EMBED_MODEL
    from .loader import load_pdf
    from .llm import get_embeddings
    from .pdf_text import (
        build_vocabulary,
        continues_sentence,
        join_hyphenated,
        strip_page_number,
        take_leading_captions,
    )
except ImportError:
    from config import CHUNK_SIZE, CHUNK_OVERLAP, FAISS_INDEX_PATH, GEMINI_EMBED_MODEL
    from loader import load_pdf
    from llm import get_embeddings
    from pdf_text import (
        build_vocabulary,
        continues_sentence,
        join_hyphenated,
        strip_page_number,
        take_leading_captions,
    )


def _index_exists(index_path: str) -> bool:
    return os.path.exists(os.path.join(index_path, "index.faiss")) and os.path.exists(
        os.path.join(index_path, "index.pkl")
    )


def _clean_pages(documents: List[Document]):
    """Normalize each page and repair sentences broken at page boundaries."""
    vocabulary = build_vocabulary([document.page_content for document in documents])

    pages = []
    for document in documents:
        text = join_hyphenated(strip_page_number(document.page_content), vocabulary)
        if text.strip():
            pages.append([document, text, "\n"])

    for index in range(1, len(pages)):
        previous = pages[index - 1][1]
        if not continues_sentence(previous):
            continue

        # A caption printed at the top of the page interrupts the running sentence.
        captions, body = take_leading_captions(pages[index][1])
        if captions:
            pages[index][1] = body + "\n" + "\n".join(captions)

        current = pages[index][1]
        if previous.rstrip().endswith("-") and current[:1].islower():
            left = previous.rstrip()[:-1]
            word = left.split()[-1] if left.split() else ""
            merged = word + current.split("\n", 1)[0].split(" ", 1)[0]
            if merged.lower() in vocabulary:
                pages[index - 1][1] = previous.rstrip()[:-1]
            else:
                pages[index - 1][1] = previous.rstrip()
            pages[index - 1][2] = ""

    return pages


def _join_pages(documents: List[Document]):
    """Concatenate page documents, remembering which span of text each page owns."""
    pieces = []
    spans = []
    cursor = 0

    pages = _clean_pages(documents)
    for position, (document, text, _) in enumerate(pages):
        if position:
            joiner = pages[position - 1][2]
            if joiner:
                pieces.append(joiner)
                cursor += len(joiner)
        start = cursor
        pieces.append(text)
        cursor += len(text)
        spans.append((start, cursor, document))

    return "".join(pieces), spans


def split_documents(
    documents: List[Document],
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> List[Document]:
    """Split the whole document, so a sentence crossing a page break stays in one chunk."""
    text, spans = _join_pages(documents)
    if not text:
        return []

    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    chunks = []
    search_from = 0
    for content in splitter.split_text(text):
        start = text.find(content, search_from)
        if start == -1:
            start = text.find(content)
        if start == -1:
            continue
        end = start + len(content)
        # Chunks overlap, so the next one may begin before this one ends.
        search_from = start + 1

        covered = [doc for span_start, span_end, doc in spans if span_start < end and span_end > start]
        metadata = dict(covered[0].metadata) if covered else {}
        if covered:
            metadata["pages"] = [doc.metadata.get("page") for doc in covered]

        chunks.append(Document(page_content=content, metadata=metadata))

    return chunks


def build_vectorstore(
    pdf_path: str,
    embeddings,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> FAISS:
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    documents = load_pdf(pdf_path)
    chunks = split_documents(documents, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return FAISS.from_documents(chunks, embeddings)


def get_vectorstore(
    pdf_path: str,
    index_path: str = FAISS_INDEX_PATH,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
    embedding_model: Optional[str] = None,
    rebuild: bool = False,
) -> FAISS:
    embeddings = get_embeddings(embedding_model or GEMINI_EMBED_MODEL)

    if not rebuild and _index_exists(index_path):
        return FAISS.load_local(
            index_path,
            embeddings,
            allow_dangerous_deserialization=True,
        )

    vectorstore = build_vectorstore(
        pdf_path=pdf_path,
        embeddings=embeddings,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    vectorstore.save_local(index_path)
    return vectorstore
