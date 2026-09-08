from langchain_core.documents import Document

from rag_core import ask_question, build_prompt, format_history, sources_from_docs


def test_prompt_contains_context_and_question():
    prompt = build_prompt("Some retrieved text.", "What is this?")

    assert "Some retrieved text." in prompt
    assert "What is this?" in prompt
    assert "only the context" in prompt


def test_prompt_omits_history_block_when_no_history():
    assert "Previous conversation" not in build_prompt("context", "question")


def test_prompt_marks_history_as_background_only():
    prompt = build_prompt("context", "question", history_text="User: hi\nAI: hello")

    assert "Previous conversation" in prompt
    assert "background only" in prompt
    # The document context must still be the stated source of truth.
    assert prompt.index("Previous conversation") < prompt.index("Context:")


def test_sources_convert_page_index_to_page_number():
    docs = [Document(page_content="text", metadata={"page": 0})]

    assert sources_from_docs(docs)[0]["page"] == 1


def test_sources_handle_missing_page_metadata():
    docs = [Document(page_content="text", metadata={})]

    assert sources_from_docs(docs)[0]["page"] is None


def test_sources_truncate_excerpt():
    docs = [Document(page_content="x" * 500, metadata={"page": 1})]

    assert len(sources_from_docs(docs, max_chars=50)[0]["excerpt"]) == 50


def test_format_history_labels_roles():
    history = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
    ]

    assert format_history(history) == "User: hi\nAI: hello"


def test_format_history_applies_limit():
    history = [{"role": "user", "content": str(i)} for i in range(6)]

    assert format_history(history, limit=2) == "User: 4\nUser: 5"


def test_ask_question_returns_answer_and_docs(fake_vectorstore, fake_llm):
    answer, docs = ask_question("What is this?", fake_vectorstore, fake_llm, k=2)

    assert answer == "A fake answer."
    assert len(docs) == 2
    assert fake_vectorstore.last_k == 2
    assert fake_vectorstore.last_query == "What is this?"


def test_ask_question_puts_retrieved_text_in_prompt(fake_vectorstore, fake_llm):
    ask_question("What is this?", fake_vectorstore, fake_llm, k=2)

    assert "The Transformer uses self-attention." in fake_llm.prompts[0]


def test_ask_question_includes_history_when_given(fake_vectorstore, fake_llm):
    ask_question("And then?", fake_vectorstore, fake_llm, k=1, history_text="User: hi\nAI: hello")

    assert "User: hi" in fake_llm.prompts[0]
