"""Checks that the UI talks to the API client and never runs RAG work itself."""

import pytest
import streamlit as st

import api_client
from streamlit.testing.v1 import AppTest

APP_PATH = "src/streamlit_app.py"


class StubClient:
    instances = []

    def __init__(self, *args, **kwargs):
        self.base_url = "http://stub"
        self.calls = []
        StubClient.instances.append(self)

    def health(self):
        return True

    def ask(self, query, k=None, session_id=None):
        self.calls.append({"query": query, "k": k, "session_id": session_id})
        return {
            "answer": f"Answer to: {query}",
            "sources": [{"page": 3, "excerpt": "retrieved text"}],
            "session_id": session_id,
            "model": "gemini-2.5-flash",
        }


@pytest.fixture
def app(monkeypatch):
    # The app caches its client with st.cache_resource, which outlives one AppTest.
    st.cache_resource.clear()
    StubClient.instances = []
    monkeypatch.setattr(api_client, "RagAPIClient", StubClient)
    return AppTest.from_file(APP_PATH, default_timeout=30).run()


def last_client() -> StubClient:
    return StubClient.instances[-1]


def test_question_is_sent_through_the_api_client(app):
    app.chat_input[0].set_value("What is the Transformer?").run()

    assert last_client().calls[0]["query"] == "What is the Transformer?"


def test_answer_and_sources_come_from_the_api_response(app):
    app.chat_input[0].set_value("What is the Transformer?").run()

    rendered = [element.value for element in app.markdown]
    assert any("Answer to: What is the Transformer?" in text for text in rendered)
    assert any("Source 1" in text and "Page 3" in text for text in rendered)


def test_the_answering_model_is_shown(app):
    app.chat_input[0].set_value("What is the Transformer?").run()

    assert any("Answered by gemini-2.5-flash" in c.value for c in app.caption)


def test_same_session_id_is_reused_for_follow_ups(app):
    app.chat_input[0].set_value("first").run()
    app.chat_input[0].set_value("second").run()

    session_ids = {call["session_id"] for call in last_client().calls}
    assert len(session_ids) == 1
    assert all(session_ids)


def test_new_chat_starts_a_new_session(app):
    app.chat_input[0].set_value("first").run()
    first_session = last_client().calls[0]["session_id"]

    app.sidebar.button[0].click().run()
    app.chat_input[0].set_value("second").run()

    assert last_client().calls[-1]["session_id"] != first_session


def test_example_question_button_sends_that_question(app):
    example = [b for b in app.button if b.label == "What is the main idea of this paper?"]

    assert example, "expected example questions on the empty state"
    example[0].click().run()

    assert last_client().calls[0]["query"] == "What is the main idea of this paper?"


def test_examples_disappear_once_the_chat_has_started(app):
    app.chat_input[0].set_value("first question").run()

    labels = [b.label for b in app.button]
    assert "What is the main idea of this paper?" not in labels


def test_api_failure_is_shown_as_a_friendly_message(app, monkeypatch):
    def failing_ask(*args, **kwargs):
        raise api_client.APIError("Cannot reach the API at http://stub. Is the backend running?")

    monkeypatch.setattr(StubClient, "ask", failing_ask)
    app.chat_input[0].set_value("anything").run()

    assert not app.exception
    assert "Cannot reach the API" in app.error[0].value


def test_streamlit_does_not_import_the_rag_pipeline():
    source = open(APP_PATH, encoding="utf-8").read()

    for forbidden in ["rag_core", "vector_store", "get_llm", "get_embeddings", "langchain"]:
        assert forbidden not in source
