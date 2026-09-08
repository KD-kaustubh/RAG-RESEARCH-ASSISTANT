from typing import List, Optional, Tuple

from langchain_core.documents import Document

try:
    from .config import TOP_K
except ImportError:
    from config import TOP_K


def build_prompt(context: str, query: str, history_text: str = "") -> str:
    history_block = f"Previous conversation:\n{history_text}\n\n" if history_text else ""
    return (
        "Answer the question using only the context below. If the context is insufficient, "
        "say you do not know.\n\n"
        f"{history_block}"
        f"Context:\n{context}\n\n"
        f"Question: {query}\n"
        "Answer:"
    )


def ask_question(
    query: str,
    vectorstore,
    llm,
    k: int = TOP_K,
    history_text: str = "",
) -> Tuple[str, List[Document]]:
    docs = vectorstore.similarity_search(query, k=k)
    context = "\n\n".join(doc.page_content for doc in docs)
    prompt = build_prompt(context, query, history_text)
    response = llm.invoke(prompt)
    answer = response.content if hasattr(response, "content") else str(response)
    return answer, docs


def sources_from_docs(docs: List[Document], max_chars: int = 300) -> List[dict]:
    sources = []
    for doc in docs:
        page = doc.metadata.get("page")
        page_number = page + 1 if isinstance(page, int) else None
        sources.append(
            {
                "page": page_number,
                "excerpt": doc.page_content[:max_chars],
            }
        )
    return sources


def format_history(history: List[dict], limit: Optional[int] = None) -> str:
    messages = history[-limit:] if limit else history
    lines = []
    for message in messages:
        role = "User" if message.get("role") == "user" else "AI"
        lines.append(f"{role}: {message.get('content', '')}")
    return "\n".join(lines)
