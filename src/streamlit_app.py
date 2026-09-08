import uuid

import streamlit as st

try:
    from .api_client import APIError, RagAPIClient
    from .config import TOP_K
except ImportError:
    from api_client import APIError, RagAPIClient
    from config import TOP_K


st.set_page_config(page_title="RAG Research Assistant", page_icon=":page_facing_up:", layout="wide")


@st.cache_resource(show_spinner=False)
def get_client() -> RagAPIClient:
    return RagAPIClient()


def _start_new_chat() -> None:
    """Drop the UI history and the server side conversation with it."""
    st.session_state.messages = []
    st.session_state.session_id = uuid.uuid4().hex


def render_sources(sources) -> None:
    if not sources:
        return

    with st.expander("Sources", expanded=True):
        for index, source in enumerate(sources, start=1):
            page = source.get("page")
            page_label = f"page {page}" if page else "page unknown"
            st.markdown(f"**Source {index} - {page_label}**")
            st.caption(source.get("excerpt", ""))


def main() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "session_id" not in st.session_state:
        st.session_state.session_id = uuid.uuid4().hex

    client = get_client()

    st.title("RAG Research Assistant")

    with st.sidebar:
        st.header("Document")
        uploaded = st.file_uploader("Upload research paper", type=["pdf"])
        if uploaded is not None and st.button("Index this paper", use_container_width=True):
            with st.spinner("Indexing the document..."):
                try:
                    result = client.upload(uploaded.name, uploaded.getvalue())
                except APIError as exc:
                    st.error(str(exc))
                else:
                    st.session_state.document = result
                    _start_new_chat()
                    st.rerun()

        active = st.session_state.get("document")
        if active:
            st.success(f"Active paper: {active['filename']}")
            st.caption(f"{active['chunks']} chunks indexed. Questions now refer to this paper.")
        else:
            st.caption("Using the default paper. Upload a PDF to replace it.")

        st.divider()
        st.header("Retrieval")
        top_k = st.slider("Sources", min_value=1, max_value=8, value=TOP_K)

        if st.button("New chat", use_container_width=True):
            _start_new_chat()
            st.rerun()

        st.divider()
        st.caption(f"API: {client.base_url}")
        if client.health():
            st.caption("Backend status: online")
        else:
            st.warning("Backend is not reachable. Start the API and reload this page.")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant":
                render_sources(message.get("sources"))

    query = st.chat_input("Ask a question about the document")
    if not query:
        return

    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Searching and generating answer..."):
            try:
                result = client.ask(
                    query,
                    k=top_k,
                    session_id=st.session_state.session_id,
                )
            except APIError as exc:
                st.error(str(exc))
                return

        st.markdown(result["answer"])
        render_sources(result["sources"])

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": result["answer"],
            "sources": result["sources"],
        }
    )


if __name__ == "__main__":
    main()
