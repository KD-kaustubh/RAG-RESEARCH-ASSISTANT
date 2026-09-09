"""Fallback tests. No provider is contacted; both models are stand-ins."""

import pytest

import llm as llm_module
from llm import FallbackLLM, get_chat_model


class Reply:
    def __init__(self, content):
        self.content = content


class StubModel:
    def __init__(self, answer="answer", error=None):
        self.answer = answer
        self.error = error
        self.calls = 0

    def invoke(self, prompt):
        self.calls += 1
        if self.error:
            raise self.error
        return Reply(self.answer)


def test_primary_answers_when_it_works():
    primary, backup = StubModel("from gemini"), StubModel("from groq")

    result = FallbackLLM(primary, backup).invoke("question")

    assert result.content == "from gemini"
    assert backup.calls == 0


def test_backup_answers_when_the_primary_fails():
    primary = StubModel(error=RuntimeError("429 RESOURCE_EXHAUSTED"))
    backup = StubModel("from groq")

    result = FallbackLLM(primary, backup).invoke("question")

    assert result.content == "from groq"
    assert backup.calls == 1


def test_last_provider_records_which_model_answered():
    fallback = FallbackLLM(StubModel("ok"), StubModel("backup"))
    fallback.invoke("question")
    assert fallback.last_provider == "primary"

    fallback = FallbackLLM(StubModel(error=RuntimeError("quota")), StubModel("backup"))
    fallback.invoke("question")
    assert fallback.last_provider == "backup"


def test_backup_failure_is_raised_to_the_caller():
    primary = StubModel(error=RuntimeError("quota exceeded"))
    backup = StubModel(error=RuntimeError("groq is down"))

    with pytest.raises(RuntimeError, match="groq is down"):
        FallbackLLM(primary, backup).invoke("question")


def test_chat_model_is_plain_gemini_without_a_groq_key(monkeypatch):
    primary = StubModel()
    monkeypatch.setattr(llm_module, "GROQ_API_KEY", None)
    monkeypatch.setattr(llm_module, "get_llm", lambda **kwargs: primary)

    assert get_chat_model() is primary


def test_chat_model_wraps_gemini_when_a_groq_key_is_set(monkeypatch):
    primary, backup = StubModel(), StubModel()
    monkeypatch.setattr(llm_module, "GROQ_API_KEY", "test-key")
    monkeypatch.setattr(llm_module, "get_llm", lambda **kwargs: primary)
    monkeypatch.setattr(llm_module, "get_groq_llm", lambda **kwargs: backup)

    model = get_chat_model()

    assert isinstance(model, FallbackLLM)
    assert model.primary is primary
    assert model.backup is backup


def test_gemini_is_used_when_the_backup_cannot_be_created(monkeypatch):
    primary = StubModel()
    monkeypatch.setattr(llm_module, "GROQ_API_KEY", "test-key")
    monkeypatch.setattr(llm_module, "get_llm", lambda **kwargs: primary)

    def broken(**kwargs):
        raise RuntimeError("bad groq configuration")

    monkeypatch.setattr(llm_module, "get_groq_llm", broken)

    assert get_chat_model() is primary


def test_groq_model_requires_a_key(monkeypatch):
    monkeypatch.setattr(llm_module, "GROQ_API_KEY", None)

    with pytest.raises(RuntimeError, match="GROQ_API_KEY"):
        llm_module.get_groq_llm()


def test_fallback_answer_still_flows_through_the_rag_pipeline(fake_vectorstore):
    """A backup answer must return a plain string with sources, like any other."""
    from rag_core import ask_question

    model = FallbackLLM(StubModel(error=RuntimeError("quota")), StubModel("Groq answer."))
    answer, docs = ask_question("What is this?", fake_vectorstore, model, k=2)

    assert answer == "Groq answer."
    assert len(docs) == 2
