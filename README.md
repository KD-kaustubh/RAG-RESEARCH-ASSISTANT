# RAG Research Assistant

A Retrieval Augmented Generation assistant for asking questions about PDF research papers. It uses LangChain, FAISS, Google Gemini, Streamlit, FastAPI, and a CLI.

## Architecture

The Streamlit UI is a client of the API. It never runs retrieval or calls Gemini
itself. The CLI still uses the RAG core directly.

```text
Streamlit UI --HTTP--> FastAPI --> RAG core --> FAISS / Gemini
   :8501                 :8000
```

In Docker these are two services built from one image: the API owns the RAG
pipeline and the FAISS index, and the UI only makes HTTP calls.

## Features

- Streamlit chat UI that talks to the API over HTTP
- Source excerpts with page numbers
- CLI single-question mode
- CLI interactive chat mode
- FastAPI `/ask` and `/health` endpoints
- FAISS index caching for the default PDF
- Dockerfile and Render blueprint for deployment

## Project Structure

```text
rag-research-assistant/
|-- src/
|   |-- api_client.py
|   |-- config.py
|   |-- loader.py
|   |-- llm.py
|   |-- main.py
|   |-- rag_assistant.py
|   |-- rag_core.py
|   |-- session_store.py
|   |-- streamlit_app.py
|   `-- vector_store.py
|-- tests/
|-- .env.example
|-- .dockerignore
|-- Dockerfile
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

The UI answers questions about the document configured by `PDF_PATH` on the
server. PDF upload was removed from the UI in this step: uploading previously
built a vector store inside Streamlit, which no longer fits the client/server
split. Restoring it needs a document endpoint on the API (upload a PDF, get a
document id back, then pass that id to `/ask`), which is not implemented yet.

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

## Tests

Install the development dependencies and run the suite:

```powershell
pip install -r requirements-dev.txt
pytest -q
```

The tests cover prompt building, source/page metadata, history formatting,
chunking and FAISS persistence, and the API (`/health`, `/ask`, validation
errors, failure handling and session memory).

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
| `PDF_PATH` | `data/paper.pdf` | Default PDF path |
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

## Troubleshooting

**Missing `GOOGLE_API_KEY`**

Create `.env` from `.env.example` and set `GOOGLE_API_KEY`.

**PDF not found**

Set `PDF_PATH=paper.pdf` or place a PDF at `data/paper.pdf` on the machine running the API.

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
