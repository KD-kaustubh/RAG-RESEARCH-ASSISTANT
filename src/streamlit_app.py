import hashlib
import os
import tempfile
from pathlib import Path

import streamlit as st

try:
    from .config import (
        CHUNK_OVERLAP,
        CHUNK_SIZE,
        FAISS_INDEX_PATH,
        GEMINI_CHAT_MODEL,
        GEMINI_EMBED_MODEL,
        PDF_PATH,
        TOP_K,
    )
    from .llm import get_llm
    from .rag_core import ask_question, format_history, sources_from_docs
    from .vector_store import get_vectorstore
except ImportError:
    from config import (
        CHUNK_OVERLAP,
        CHUNK_SIZE,
        FAISS_INDEX_PATH,
        GEMINI_CHAT_MODEL,
        GEMINI_EMBED_MODEL,
        PDF_PATH,
        TOP_K,
    )
    from llm import get_llm
    from rag_core import ask_question, format_history, sources_from_docs
    from vector_store import get_vectorstore


st.set_page_config(page_title="RAG Research Assistant", page_icon=":page_facing_up:", layout="wide")


def _file_hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()[:16]


def _reset_chat() -> None:
    st.session_state.messages = []


@st.cache_resource(show_spinner=False)
def load_default_vectorstore(pdf_path: str, index_path: str, chunk_size: int, chunk_overlap: int):
    return get_vectorstore(
        pdf_path=pdf_path,
        index_path=index_path,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )


@st.cache_resource(show_spinner=False)
def load_llm():
    return get_llm()


def load_uploaded_vectorstore(uploaded_file, chunk_size: int, chunk_overlap: int):
    file_bytes = uploaded_file.getvalue()
    upload_hash = _file_hash(file_bytes)
    upload_signature = (upload_hash, chunk_size, chunk_overlap)

    if (
        st.session_state.get("upload_signature") == upload_signature
        and "uploaded_vectorstore" in st.session_state
    ):
        return st.session_state.uploaded_vectorstore

    temp_dir = tempfile.mkdtemp(prefix="rag_upload_")
    safe_name = Path(uploaded_file.name).name
    pdf_path = os.path.join(temp_dir, safe_name)
    index_path = os.path.join(temp_dir, "faiss_index")

    with open(pdf_path, "wb") as pdf_file:
        pdf_file.write(file_bytes)

    vectorstore = get_vectorstore(
        pdf_path=pdf_path,
        index_path=index_path,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        rebuild=True,
    )

    st.session_state.upload_signature = upload_signature
    st.session_state.uploaded_vectorstore = vectorstore
    st.session_state.uploaded_pdf_name = safe_name
    _reset_chat()
    return vectorstore


def render_sources(docs) -> None:
    sources = sources_from_docs(docs)
    if not sources:
        return

    with st.expander("Sources", expanded=True):
        for index, source in enumerate(sources, start=1):
            page = f"page {source['page']}" if source["page"] else "page unknown"
            st.markdown(f"**Source {index} - {page}**")
            st.caption(source["excerpt"])


def resolve_default_pdf() -> str:
    configured_path = Path(PDF_PATH)
    if configured_path.exists():
        return str(configured_path)

    fallback_path = Path("paper.pdf")
    if fallback_path.exists():
        return str(fallback_path)

    return str(configured_path)


def main() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = []

    st.title("RAG Research Assistant")

    with st.sidebar:
        st.header("Document")
        uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])

        st.header("Retrieval")
        top_k = st.slider("Sources", min_value=1, max_value=8, value=TOP_K)
        chunk_size = st.number_input("Chunk size", min_value=200, max_value=2000, value=CHUNK_SIZE, step=100)
        chunk_overlap = st.number_input(
            "Chunk overlap",
            min_value=0,
            max_value=500,
            value=CHUNK_OVERLAP,
            step=25,
        )

        if st.button("Reset chat", use_container_width=True):
            _reset_chat()

        st.divider()
        st.caption(f"Chat model: {GEMINI_CHAT_MODEL}")
        st.caption(f"Embedding model: {GEMINI_EMBED_MODEL}")

    try:
        with st.spinner("Preparing document index..."):
            if uploaded_file:
                vectorstore = load_uploaded_vectorstore(uploaded_file, chunk_size, chunk_overlap)
                document_name = st.session_state.get("uploaded_pdf_name", uploaded_file.name)
            else:
                default_pdf = resolve_default_pdf()
                if not Path(default_pdf).exists():
                    st.error(f"Default PDF not found: {default_pdf}. Upload a PDF or set PDF_PATH.")
                    return
                vectorstore = load_default_vectorstore(default_pdf, FAISS_INDEX_PATH, chunk_size, chunk_overlap)
                document_name = Path(default_pdf).name

            llm = load_llm()
    except Exception as exc:
        st.error(f"Could not initialize the assistant: {exc}")
        return

    st.caption(f"Current document: {document_name}")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant" and message.get("docs"):
                render_sources(message["docs"])

    query = st.chat_input("Ask a question about the document")
    if not query:
        return

    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    history_text = format_history(st.session_state.messages[:-1], limit=10)

    with st.chat_message("assistant"):
        with st.spinner("Searching and generating answer..."):
            try:
                answer, docs = ask_question(query, vectorstore, llm, k=top_k, history_text=history_text)
            except Exception as exc:
                st.error(f"Could not answer the question: {exc}")
                return

        st.markdown(answer)
        render_sources(docs)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "docs": docs,
        }
    )


if __name__ == "__main__":
    main()
