# RAG Research Assistant

## Overview

Upload a research paper, ask questions about it, and get answers that are grounded
in the document itself and cited back to the page they came from. If the paper does
not contain the answer, the assistant says so rather than inventing one.

The project ships with the "Attention Is All You Need" paper so it works the moment
you start it, and any PDF you upload replaces it as the active document.

## Key Features

- PDF research paper upload through the UI
- PDF text cleaning that repairs page numbers, interrupted captions and words broken across lines
- Cross-page chunking, so a sentence split by a page break stays whole
- Gemini embeddings and FAISS vector search
- Grounded Gemini answers with page-level source citations
- Optional Groq fallback so answers keep working when Gemini hits its daily cap
- Session-based conversational memory, so follow-up questions resolve "it" and "its"
- FastAPI backend and a Streamlit frontend that talks to it over HTTP
- CLI for single questions and interactive chat
- Docker Compose setup running the API and UI as separate services
- 129 automated tests that need no API key or network access
- Render deployment blueprint

## Architecture

The UI is purely an API client. It never imports the RAG code, touches FAISS, or
calls Gemini; the API owns all of that. The CLI talks to the RAG core directly.

```text
Streamlit UI  --HTTP-->  FastAPI  -->  RAG core  -->  FAISS + Gemini
    :8501                 :8000
```

## RAG Pipeline

```text
PDF
  -> text extraction (PyPDFLoader)
  -> text normalization (page numbers, captions, hyphenated line breaks)
  -> cross-page chunking (RecursiveCharacterTextSplitter, page metadata preserved)
  -> embeddings (Gemini)
  -> FAISS index
  -> similarity retrieval (top k)
  -> prompt with retrieved context + bounded chat history
  -> Gemini
  -> grounded answer + page citations
```

## Answer Fallback

Gemini's free tier allows only a small number of requests per day, which is enough
to stop a live demo mid-conversation. Setting `GROQ_API_KEY` enables a backup: if
the Gemini call fails for any reason, the same prompt and the same retrieved
context are sent to Groq instead, and the answer comes back in the usual shape
with its citations. Which provider answered is recorded in the server log.

Retrieval is unaffected. **Embeddings always use Gemini**, because the FAISS index
is built from Gemini vectors and mixing providers would invalidate it. The backup
covers answer generation only, so an exhausted Gemini *embedding* quota still
blocks indexing a newly uploaded PDF.

Leave `GROQ_API_KEY` unset and the assistant runs on Gemini alone, exactly as before.

## Tech Stack

Python 3.12, FastAPI, Streamlit, LangChain, FAISS (`faiss-cpu`), Google Gemini
(`gemini-2.5-flash` for chat, `gemini-embedding-001` for embeddings), Groq as an
optional answer fallback, pypdf, Pydantic, pytest, Docker Compose, Render.

## Project Structure

```text
rag-research-assistant/
|-- src/
|   |-- api_client.py      HTTP client the UI uses to reach the API
|   |-- config.py          environment-driven settings
|   |-- loader.py          PDF loading
|   |-- pdf_text.py        extraction cleanup
|   |-- vector_store.py    chunking, FAISS build/load
|   |-- llm.py             Gemini chat and embedding models
|   |-- rag_core.py        retrieval, prompt, citations
|   |-- session_store.py   bounded in-memory chat history
|   |-- main.py            FastAPI app
|   |-- streamlit_app.py   chat UI
|   `-- rag_assistant.py   CLI
|-- tests/
|-- .env.example
|-- .dockerignore
|-- Dockerfile
|-- docker-compose.yml
|-- README.md
|-- render.yaml
|-- requirements.txt
`-- paper.pdf
```

## Setup

Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\activate
```

Install dependencies:

```powershell
pip install -r requirements.txt
```

Create your environment file:

```powershell
copy .env.example .env
```

Set your Gemini API key in `.env`:

```env
GOOGLE_API_KEY=your_google_api_key_here
```

The default PDF path is controlled by `PDF_PATH`. The included demo setup uses:

```env
PDF_PATH=paper.pdf
```

## Run Streamlit

The UI is an API client, so start the backend first:

```powershell
uvicorn src.main:app --reload
```

Then, in a second terminal:

```powershell
streamlit run src\streamlit_app.py
```

Open the local URL printed by Streamlit, usually:

```text
http://localhost:8501
```

Point the UI at a different backend with `API_BASE_URL`:

```env
API_BASE_URL=http://localhost:8000
```

The sidebar shows whether the backend is reachable. Each browser session gets its
own conversation, and **New chat** starts a fresh one.

The sidebar shows the active paper. Upload a PDF there to replace it: the file is
sent to the API, which re-indexes it and makes it the document every later
question is answered from.

## Run CLI

Single question:

```powershell
python src\rag_assistant.py --query "What is the main idea of the paper?"
```

Interactive chat mode:

```powershell
python src\rag_assistant.py --interactive
```

Useful CLI options:

```powershell
python src\rag_assistant.py --pdf paper.pdf --k 3 --rebuild-index
```

## Run FastAPI

```powershell
uvicorn src.main:app --reload
```

Open the API docs:

```text
http://127.0.0.1:8000/docs
```

Health check:

```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:8000/health"
```

Example request:

```powershell
Invoke-RestMethod `
  -Uri "http://127.0.0.1:8000/ask" `
  -Method Post `
  -ContentType "application/json" `
  -Body '{"query":"What is the Transformer?","k":2}'
```

Pass a `session_id` to ask follow-up questions. History is kept in memory per
session and is lost when the server restarts:

```powershell
Invoke-RestMethod `
  -Uri "http://127.0.0.1:8000/ask" `
  -Method Post `
  -ContentType "application/json" `
  -Body '{"query":"What are its main components?","session_id":"demo-1"}'
```

Example response shape:

```json
{
  "answer": "The Transformer is a model architecture...",
  "sources": [
    {
      "page": 2,
      "excerpt": "The Transformer follows this overall architecture..."
    }
  ]
}
```

## API

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/health` | Liveness check. Returns `{"status": "ok"}` and never calls Gemini, so provider quota cannot mark the service unhealthy. |
| `POST` | `/ask` | Ask a question. Body: `query` (required), `k` (1-20), `session_id` (optional). Returns the answer, page-cited sources, and the session id. |
| `POST` | `/upload` | Replace the active document with a PDF (multipart `file`). Rebuilds the index and returns `{"status", "filename", "chunks"}`. |

Uploading replaces the active document and clears existing conversations, so a
new paper never inherits history about the previous one.

Errors use plain JSON with a `detail` message and never expose tracebacks,
internal paths, or credentials:

| Status | Meaning |
| --- | --- |
| `400` | Not a PDF, empty file, or a PDF that cannot be read |
| `413` | Upload above `MAX_UPLOAD_MB` |
| `422` | Invalid request body |
| `502` | Gemini failed or was rate limited |
| `503` | Missing API key or the document could not be loaded |

Upload example:

```powershell
curl.exe -X POST "http://127.0.0.1:8000/upload" -F "file=@paper.pdf;type=application/pdf"
```

## Tests

Install the development dependencies and run the suite:

```powershell
pip install -r requirements-dev.txt
pytest -q
```

129 tests cover prompt building, source and page metadata, history formatting,
PDF cleanup, cross-page chunking, FAISS persistence, upload validation and
indexing, the API (`/health`, `/ask`, `/upload`, validation and failure paths),
the UI's use of the API client, and a fixed retrieval evaluation set for the
bundled paper.

Gemini calls are replaced with test doubles, so the suite needs no API key, no
network access and no prebuilt FAISS index.

## Docker

Both services share one image. `docker-compose.yml` runs the API on port 8000 and
the UI on port 8501, and the UI reaches the API at `http://api:8000` on the
compose network.

Make sure `.env` exists with your `GOOGLE_API_KEY`, then:

```powershell
docker compose up --build
```

Open the UI:

```text
http://localhost:8501
```

The API is available separately:

```text
http://localhost:8000/health
http://localhost:8000/docs
```

Stop everything with:

```powershell
docker compose down
```

The API container builds the FAISS index on its first question and keeps it in a
named volume, so later restarts reuse it instead of re-embedding the PDF. The UI
never builds an index.

To run a single service from the image, override the command. The image defaults
to the API:

```powershell
docker run --rm -p 8000:8000 --env-file .env rag-research-assistant
```

## Deploy On Render

`render.yaml` defines two Docker services built from the same repository:

| Service | Runs | Purpose |
| --- | --- | --- |
| `rag-api` | `uvicorn src.main:app` | RAG pipeline, FAISS index, `/ask` and `/health` |
| `rag-ui` | `streamlit run src/streamlit_app.py` | Chat UI, talks to `rag-api` over HTTP |

Both bind to Render's `$PORT`. The API uses `/health` as its health check, which
reports process liveness only and never calls Gemini, so quota problems cannot
mark the service unhealthy.

1. Push the repo to GitHub.
2. In Render, create a new Blueprint from the repo.
3. On `rag-api`, set `GOOGLE_API_KEY`. Deploy it first and confirm
   `https://<rag-api>.onrender.com/health` returns `{"status": "ok"}`.
4. On `rag-ui`, set `API_BASE_URL` to the backend URL, including the scheme,
   for example `https://rag-api.onrender.com`.
5. Deploy `rag-ui` and open its URL. The sidebar shows the backend status.

`ALLOWED_ORIGINS` is only needed if a browser calls the API directly; the UI
calls it server side, so CORS does not apply to normal use. Leaving it unset in
Render means no browser origin is allowed, which is the safe default.

Secrets are never committed. `GOOGLE_API_KEY` and `API_BASE_URL` are marked
`sync: false`, so Render asks for them at deploy time.

### Free tier notes

- Free services sleep when idle, so the first request after a pause is slow.
- Free instances have no persistent disk, so the API rebuilds the FAISS index
  after each restart. That costs embedding calls and makes the first question
  slower; `API_TIMEOUT` is set to 120 seconds on the UI to allow for it.
- The Gemini free tier caps daily generation requests, which surfaces as a
  clean "Failed to generate an answer" message rather than a crash.

## Configuration

All configuration is read from `.env` or environment variables.

| Variable | Default | Purpose |
| --- | --- | --- |
| `GOOGLE_API_KEY` | required | Gemini API key |
| `GEMINI_EMBED_MODEL` | `gemini-embedding-001` | Embedding model |
| `GEMINI_CHAT_MODEL` | `gemini-2.5-flash` | Chat model |
| `GROQ_API_KEY` | unset | Enables the Groq backup for answers when set |
| `GROQ_CHAT_MODEL` | `openai/gpt-oss-120b` | Backup chat model |
| `PDF_PATH` | `paper.pdf` | Default PDF used until something is uploaded |
| `QUERY` | `What is the main idea of the paper?` | Default CLI question |
| `CHUNK_SIZE` | `500` | PDF chunk size |
| `CHUNK_OVERLAP` | `50` | PDF chunk overlap |
| `TOP_K` | `3` | Number of retrieved chunks |
| `FAISS_INDEX_PATH` | `faiss_index` | Saved FAISS index directory |
| `API_BASE_URL` | `http://localhost:8000` | Backend URL used by the Streamlit UI |
| `API_TIMEOUT` | `60` | Seconds the UI waits for an API answer |
| `ALLOWED_ORIGINS` | `http://localhost:8501,http://127.0.0.1:8501` | Comma separated CORS origins for the API |
| `MAX_HISTORY_MESSAGES` | `10` | Messages kept per API session |
| `MAX_SESSIONS` | `100` | Sessions kept in memory before the oldest is dropped |
| `UPLOAD_DIR` | `uploads` | Where the active uploaded PDF is stored |
| `MAX_UPLOAD_MB` | `10` | Largest accepted upload |

## Troubleshooting

**Missing `GOOGLE_API_KEY`**

Create `.env` from `.env.example` and set `GOOGLE_API_KEY`.

**PDF not found**

Point `PDF_PATH` at a PDF on the machine running the API, or upload one through the UI.

**Gemini model not found**

Use the current defaults:

```env
GEMINI_EMBED_MODEL=gemini-embedding-001
GEMINI_CHAT_MODEL=gemini-2.5-flash
```

**Windows console encoding errors**

The CLI configures UTF-8 output automatically. If your terminal still has issues, run:

```powershell
chcp 65001
```

**Rebuild FAISS index**

```powershell
python src\rag_assistant.py --rebuild-index
```

## Limitations

- One active document at a time. Uploading a paper replaces the previous one; this
  is deliberate, since document management is not what the project sets out to show.
- Chat history lives in the API process, so it is lost on restart and is not shared
  across multiple workers.
- Answer throughput depends on the Gemini plan in use. On the free tier the daily
  request cap is easy to reach; configure `GROQ_API_KEY` so answers fall back to
  Groq instead of failing. Embeddings have no fallback and stay on Gemini.
- Free Render instances sleep when idle and have no persistent disk, so the first
  request after a pause is slow and the index is rebuilt after a restart.
- Retrieval is plain vector similarity. Most questions about the bundled paper are
  answered well, but a broadly phrased one can still miss the best passage.

## Design Philosophy

The project keeps the architecture small on purpose. Every part earns its place by
demonstrating a step of the RAG pipeline — ingestion, cleaning, chunking,
embeddings, vector search, grounded generation, citations, and conversation state —
with a clean API/UI split, tests, and a working container setup around it. Adding a
database, a queue, or object storage would grow the diagram without teaching
anything more about retrieval-augmented generation, so they are left out.
