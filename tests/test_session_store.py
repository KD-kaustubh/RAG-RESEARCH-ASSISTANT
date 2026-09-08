from session_store import SessionStore


def test_unknown_session_has_empty_history():
    assert SessionStore().get_history("missing") == []


def test_turn_is_stored_as_user_and_assistant_messages():
    store = SessionStore()
    store.add_turn("a", "question", "answer")

    assert store.get_history("a") == [
        {"role": "user", "content": "question"},
        {"role": "assistant", "content": "answer"},
    ]


def test_history_is_bounded_and_keeps_newest():
    store = SessionStore(max_messages=4)
    for i in range(5):
        store.add_turn("a", f"q{i}", f"a{i}")

    history = store.get_history("a")
    assert len(history) == 4
    assert [message["content"] for message in history] == ["q3", "a3", "q4", "a4"]


def test_sessions_are_isolated():
    store = SessionStore()
    store.add_turn("a", "question a", "answer a")
    store.add_turn("b", "question b", "answer b")

    assert [m["content"] for m in store.get_history("a")] == ["question a", "answer a"]
    assert [m["content"] for m in store.get_history("b")] == ["question b", "answer b"]


def test_oldest_session_is_evicted_when_full():
    store = SessionStore(max_sessions=2)
    store.add_turn("a", "q", "a")
    store.add_turn("b", "q", "a")
    store.add_turn("c", "q", "a")

    assert store.session_count() == 2
    assert store.get_history("a") == []
    assert store.get_history("c") != []


def test_recently_used_session_survives_eviction():
    store = SessionStore(max_sessions=2)
    store.add_turn("a", "q", "a")
    store.add_turn("b", "q", "a")
    store.get_history("a")
    store.add_turn("c", "q", "a")

    assert store.get_history("a") != []
    assert store.get_history("b") == []


def test_returned_history_is_a_copy():
    store = SessionStore()
    store.add_turn("a", "q", "a")

    store.get_history("a").append({"role": "user", "content": "injected"})

    assert len(store.get_history("a")) == 2


def test_clear_removes_session():
    store = SessionStore()
    store.add_turn("a", "q", "a")
    store.clear("a")

    assert store.get_history("a") == []
