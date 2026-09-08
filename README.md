# RAG Research Assistant

A Retrieval Augmented Generation assistant for asking questions about PDF research papers. It uses LangChain, FAISS, Google Gemini, Streamlit, FastAPI, and a CLI.

## Features

- Streamlit chat UI for PDF question answering
- Upload a PDF or use the configured default document
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
|   |-- config.py
|   |-- loader.py
|   |-- llm.py
|   |-- main.py
|   |-- rag_assistant.py
|   |-- rag_core.py
|   |-- streamlit_app.py
|   `-- vector_store.py
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

```powershell
streamlit run src\streamlit_app.py
```

Open the local URL printed by Streamlit, usually:

```text
http://localhost:8501
```

The UI can use the default PDF or a PDF uploaded in the sidebar. Uploaded PDFs are indexed temporarily for the current Streamlit session.

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

## Docker

Build the image:

```powershell
docker build -t rag-research-assistant .
```

Run the Streamlit app:

```powershell
docker run --rm -p 8501:8501 --env-file .env rag-research-assistant
```

Open:

```text
http://localhost:8501
```

To run the FastAPI server from the same image:

```powershell
docker run --rm -p 8000:8000 --env-file .env rag-research-assistant `
  python -m uvicorn src.main:app --host 0.0.0.0 --port 8000
```

## Deploy On Render

This repo includes `render.yaml` for a Docker web service. The default deployed service runs Streamlit.

1. Push the repo to GitHub.
2. In Render, create a new Blueprint from the GitHub repo.
3. Set the secret environment variable `GOOGLE_API_KEY`.
4. Deploy the service.

Optional Render environment variables:

```env
GEMINI_EMBED_MODEL=gemini-embedding-001
GEMINI_CHAT_MODEL=gemini-2.5-flash
PDF_PATH=paper.pdf
TOP_K=3
```

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
| `ALLOWED_ORIGINS` | `http://localhost:8501,http://127.0.0.1:8501` | Comma separated CORS origins for the API |
| `MAX_HISTORY_MESSAGES` | `10` | Messages kept per API session |
| `MAX_SESSIONS` | `100` | Sessions kept in memory before the oldest is dropped |

## Troubleshooting

**Missing `GOOGLE_API_KEY`**

Create `.env` from `.env.example` and set `GOOGLE_API_KEY`.

**PDF not found**

Set `PDF_PATH=paper.pdf`, place a PDF at `data/paper.pdf`, or upload a PDF in Streamlit.

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
