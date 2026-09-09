import uuid

import streamlit as st

try:
    from .api_client import APIError, RagAPIClient
    from .config import TOP_K
except ImportError:
    from api_client import APIError, RagAPIClient
    from config import TOP_K


st.set_page_config(page_title="RAG Research Assistant", page_icon=":page_facing_up:", layout="centered")

EXAMPLE_QUESTIONS = [
    "What is the main idea of this paper?",
    "What method does it propose?",
    "What are the main results?",
]


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

    with st.expander(f"Sources ({len(sources)})", expanded=False):
        for index, source in enumerate(sources, start=1):
            page = source.get("page")
            page_label = f"Page {page}" if page else "Page unknown"
            st.markdown(f"**Source {index}** &nbsp;·&nbsp; {page_label}")
            st.caption(source.get("excerpt", ""))
            if index < len(sources):
                st.divider()


def render_answer(answer: str, sources, model=None) -> None:
    st.markdown(answer)
    render_sources(sources)
    if model:
        st.caption(f"Answered by {model}")


def render_empty_state() -> None:
    st.caption("Ask anything about the active paper, or try one of these:")
    for column, question in zip(st.columns(len(EXAMPLE_QUESTIONS)), EXAMPLE_QUESTIONS):
        if column.button(question, width="content"):
            st.session_state.pending_query = question
            st.rerun()


def main() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "session_id" not in st.session_state:
        st.session_state.session_id = uuid.uuid4().hex

    client = get_client()
    active = st.session_state.get("document")

    st.title("RAG Research Assistant")
    st.caption("Answers grounded in your PDF, with the page they came from.")

    with st.sidebar:
        st.header("Document")
        if active:
            st.markdown(f"**{active['filename']}**")
            st.caption(f"{active['chunks']} chunks indexed")
        else:
            st.markdown("**paper.pdf**")
            st.caption("Default paper")

        uploaded = st.file_uploader("Replace with your own PDF", type=["pdf"])
        if uploaded is not None and st.button("Index this paper", width="stretch"):
            with st.spinner("Indexing the document..."):
                try:
                    result = client.upload(uploaded.name, uploaded.getvalue())
                except APIError as exc:
                    st.error(str(exc))
                else:
                    st.session_state.document = result
                    _start_new_chat()
                    st.rerun()

        st.divider()
        st.header("Chat")
        if st.button("New chat", width="stretch"):
            _start_new_chat()
            st.rerun()
        top_k = st.slider(
            "Sources per answer",
            min_value=1,
            max_value=8,
            value=TOP_K,
            help="How many passages are retrieved and shown as citations.",
        )

        st.divider()
        if client.health():
            st.caption(f"Backend online · {client.base_url}")
        else:
            st.warning("Backend is not reachable. Start the API and reload this page.")

    # chat_input is pinned to the bottom of the page, so reading it first only
    # affects which elements we render, not the layout.
    query = st.chat_input("Ask a question about the paper") or st.session_state.pop(
        "pending_query", None
    )

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message["role"] == "assistant":
                render_answer(message["content"], message.get("sources"), message.get("model"))
            else:
                st.markdown(message["content"])

    if not st.session_state.messages and not query:
        render_empty_state()

    if not query:
        return

    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Searching the paper..."):
            try:
                result = client.ask(
                    query,
                    k=top_k,
                    session_id=st.session_state.session_id,
                )
            except APIError as exc:
                st.error(str(exc))
                return

        render_answer(result["answer"], result["sources"], result.get("model"))

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": result["answer"],
            "sources": result["sources"],
            "model": result.get("model"),
        }
    )


if __name__ == "__main__":
    main()
