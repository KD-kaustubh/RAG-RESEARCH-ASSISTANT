import argparse
import sys


try:
    from .config import (
        PDF_PATH,
        QUERY_DEFAULT,
        CHUNK_SIZE,
        CHUNK_OVERLAP,
        TOP_K,
        FAISS_INDEX_PATH,
    )
    from .llm import get_chat_model
    from .rag_core import ask_question, format_history, sources_from_docs
    from .vector_store import get_vectorstore
except ImportError:
    from config import (
        PDF_PATH,
        QUERY_DEFAULT,
        CHUNK_SIZE,
        CHUNK_OVERLAP,
        TOP_K,
        FAISS_INDEX_PATH,
    )
    from llm import get_chat_model
    from rag_core import ask_question, format_history, sources_from_docs
    from vector_store import get_vectorstore


def configure_output() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def print_sources(docs, max_chars: int = 300) -> None:
    for i, source in enumerate(sources_from_docs(docs, max_chars=max_chars)):
        page_label = f" (page {source['page']})" if source["page"] else ""
        print(f"\nSOURCE {i + 1}{page_label}:")
        print(source["excerpt"])
        print("-" * 50)


def run_interactive(vectorstore, llm, k: int) -> None:
    chat_history = []

    while True:
        query = input("You: ").strip()
        if not query:
            continue
        if query.lower() in {"exit", "quit"}:
            break

        history_text = format_history(chat_history)
        answer, docs = ask_question(query, vectorstore, llm, k, history_text)
        print_sources(docs)
        print(f"\nAI: {answer}\n")

        chat_history.append({"role": "user", "content": query})
        chat_history.append({"role": "assistant", "content": answer})


def main() -> None:
    configure_output()

    parser = argparse.ArgumentParser(description="RAG assistant with sources and chat memory.")
    parser.add_argument("--pdf", help="Path to PDF file")
    parser.add_argument("--query", help="Question to ask")
    parser.add_argument("--k", type=int, default=TOP_K, help="Number of chunks to retrieve")
    parser.add_argument("--chunk-size", type=int, default=CHUNK_SIZE, help="Chunk size in characters")
    parser.add_argument("--chunk-overlap", type=int, default=CHUNK_OVERLAP, help="Chunk overlap in characters")
    parser.add_argument("--index-path", default=FAISS_INDEX_PATH, help="FAISS index directory")
    parser.add_argument("--rebuild-index", action="store_true", help="Rebuild FAISS index")
    parser.add_argument("--interactive", action="store_true", help="Run in chat mode")
    args = parser.parse_args()

    pdf_path = args.pdf or PDF_PATH

    vectorstore = get_vectorstore(
        pdf_path=pdf_path,
        index_path=args.index_path,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        rebuild=args.rebuild_index,
    )

    llm = get_chat_model()

    if args.interactive:
        run_interactive(vectorstore, llm, k=args.k)
        return

    query = args.query or QUERY_DEFAULT
    answer, docs = ask_question(query, vectorstore, llm, args.k)
    print_sources(docs)
    print(f"\nAI: {answer}")


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, FileNotFoundError) as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(1)
