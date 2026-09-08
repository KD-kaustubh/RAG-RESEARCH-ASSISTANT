"""Shared test doubles.

The suite never calls Gemini and never touches the production FAISS index. The
fake embedding model below is deterministic so retrieval results are stable.
"""

from typing import List

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings


class FakeEmbeddings(Embeddings):
    """Bag-of-characters embedding, enough for FAISS to build a real index."""

    dimension = 16

    def _embed(self, text: str) -> List[float]:
        vector = [0.0] * self.dimension
        for word in text.lower().split():
            vector[hash(word) % self.dimension] += 1.0
        norm = sum(value * value for value in vector) ** 0.5
        return [value / norm for value in vector] if norm else [1.0] + [0.0] * (self.dimension - 1)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._embed(text)


class FakeLLM:
    """Records the prompts it receives so tests can assert on them."""

    def __init__(self, answer: str = "A fake answer.", error: Exception = None):
        self.answer = answer
        self.error = error
        self.prompts: List[str] = []

    def invoke(self, prompt: str):
        self.prompts.append(prompt)
        if self.error:
            raise self.error

        class Response:
            content = self.answer

        return Response()


class FakeVectorStore:
    """Returns canned documents and records the k it was asked for."""

    def __init__(self, docs: List[Document] = None):
        self.docs = docs if docs is not None else [
            Document(page_content="The Transformer uses self-attention.", metadata={"page": 0}),
            Document(page_content="It has an encoder and a decoder.", metadata={"page": 4}),
        ]
        self.last_k = None
        self.last_query = None

    def similarity_search(self, query: str, k: int = 3):
        self.last_query = query
        self.last_k = k
        return self.docs[:k]


@pytest.fixture
def fake_embeddings():
    return FakeEmbeddings()


@pytest.fixture
def fake_llm():
    return FakeLLM()


@pytest.fixture
def fake_vectorstore():
    return FakeVectorStore()
