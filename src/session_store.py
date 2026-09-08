"""In-process conversation memory for the API.

History is kept in memory only and is lost when the process restarts. Growth is
bounded twice: each session keeps at most the last `max_messages` messages, and
the store keeps at most `max_sessions` sessions, evicting the least recently
used one first.
"""

import threading
from collections import OrderedDict
from typing import Dict, List

try:
    from .config import MAX_HISTORY_MESSAGES, MAX_SESSIONS
except ImportError:
    from config import MAX_HISTORY_MESSAGES, MAX_SESSIONS


class SessionStore:
    def __init__(
        self,
        max_messages: int = MAX_HISTORY_MESSAGES,
        max_sessions: int = MAX_SESSIONS,
    ) -> None:
        self._sessions: "OrderedDict[str, List[Dict[str, str]]]" = OrderedDict()
        self._max_messages = max_messages
        self._max_sessions = max_sessions
        self._lock = threading.Lock()

    def get_history(self, session_id: str) -> List[Dict[str, str]]:
        with self._lock:
            history = self._sessions.get(session_id)
            if history is None:
                return []
            self._sessions.move_to_end(session_id)
            return list(history)

    def add_turn(self, session_id: str, query: str, answer: str) -> None:
        with self._lock:
            history = self._sessions.setdefault(session_id, [])
            history.append({"role": "user", "content": query})
            history.append({"role": "assistant", "content": answer})

            if len(history) > self._max_messages:
                del history[: len(history) - self._max_messages]

            self._sessions.move_to_end(session_id)
            while len(self._sessions) > self._max_sessions:
                self._sessions.popitem(last=False)

    def clear(self, session_id: str) -> None:
        with self._lock:
            self._sessions.pop(session_id, None)

    def session_count(self) -> int:
        with self._lock:
            return len(self._sessions)
