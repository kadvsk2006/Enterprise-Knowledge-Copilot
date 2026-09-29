# Enterprise Knowledge Copilot - Hybrid RAG System

Enterprise Knowledge Copilot is a document question-answering service built
with a hybrid retrieval-augmented generation (RAG) pipeline. Upload one PDF,
ask questions about its contents, and receive an answer grounded in retrieved
document passages with source text and page numbers.

This repository is a single-user, single-document demonstration. It keeps the
document index and conversation sessions in memory and is not a production
multi-user knowledge platform.

## Capabilities

- Extracts searchable text and page numbers from uploaded PDFs.
- Splits pages into overlapping text chunks and embeds them locally.
- Combines dense FAISS search and sparse BM25 search with Reciprocal Rank
  Fusion (RRF).
- Uses a cross-encoder to rerank retrieval candidates while preserving the
  strongest first-stage results.
- Generates grounded answers with Groq, OpenAI, or Anthropic models.
- Returns source chunks and supports both standard and server-sent event
  (SSE) streaming responses.
- Keeps up to five turns of conversation history per in-memory session.

## Architecture

```text
PDF upload
  -> pypdf text extraction with page numbers
  -> 500-character chunks with 50-character overlap
  -> normalized all-MiniLM-L6-v2 embeddings
  -> FAISS dense search + BM25 keyword search
  -> Reciprocal Rank Fusion (RRF, k=60)
  -> cross-encoder reranking
  -> selected LLM provider
  -> grounded answer and source chunks
```

Chunking is performed within each page so every source chunk retains a page
number. Embeddings use `all-MiniLM-L6-v2` (384 dimensions); normalized vectors
allow FAISS `IndexFlatIP` inner-product search to act as cosine-similarity
search. The embedding and reranker models load on first use and may download
approximately 80 MB each.

The retrieval stage combines FAISS semantic results with BM25 keyword results
using RRF. A `cross-encoder/ms-marco-MiniLM-L-6-v2` model reranks candidates.
The top `top_k - 1` first-stage results are retained, and the cross-encoder
selects the remaining result from the broader candidate list.

The LLM layer supports Groq, OpenAI, and Anthropic through the
`LLM_PROVIDER` setting. Responses are instructed to use the retrieved
document context only and to say when the answer is not present. Source
information is returned separately from the generated answer.

## Technology

| Component | Implementation |
|---|---|
| HTTP API | FastAPI and Uvicorn |
| PDF extraction | pypdf |
| Chunking and embeddings | sentence-transformers, `all-MiniLM-L6-v2` |
| Dense retrieval | FAISS `IndexFlatIP` |
| Sparse retrieval | BM25 via `rank-bm25` |
| Rank fusion | Reciprocal Rank Fusion, `k=60` |
| Reranking | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| Answer generation | Groq, OpenAI, or Anthropic |
| Configuration | Environment variables loaded from `.env` |

## Setup

Requirements: Python 3.10 or later and [`uv`](https://docs.astral.sh/uv/).

Install `uv` using its official installation instructions, then install the
project dependencies and create a local environment file:

```shell
uv sync
```

On Windows PowerShell:

```powershell
Copy-Item .env.example .env
```

On macOS or Linux:

```shell
cp .env.example .env
```

Edit `.env` and set the API key for the provider you select. Start the service:

```shell
uv run uvicorn app.main:app --reload
```

The interactive API documentation is available at
<http://localhost:8000/docs>. The included example document is
`data/sample_test_file.pdf`.

## Configuration

`LLM_PROVIDER` selects the answer-generation provider. The application defaults
to `groq` if it is unset; `.env.example` also sets it to `groq`. Only the API
key for the selected provider is needed.

| Variable | Required | Default or purpose |
|---|---|---|
| `LLM_PROVIDER` | No | Provider: `groq`, `openai`, or `anthropic`; defaults to `groq` |
| `GROQ_API_KEY` | For Groq | API key for the Groq account |
| `OPENAI_API_KEY` | For OpenAI | API key for the OpenAI account |
| `ANTHROPIC_API_KEY` | For Anthropic | API key for the Anthropic account |
| `GROQ_MODEL` | No | Defaults to `llama-3.3-70b-versatile` |
| `OPENAI_MODEL` | No | Defaults to `gpt-4o-mini` |
| `ANTHROPIC_MODEL` | No | Defaults to `claude-haiku-4-5-20251001` |

## API Usage

All endpoints are served from the FastAPI application. Interactive requests
can be sent from `/docs`.

### `GET /health`

Returns the service status and current number of indexed chunks:

```json
{"status": "ok", "chunks_indexed": 0}
```

### `POST /upload`

Accepts a PDF as multipart form data in the `file` field. Uploading a new PDF
replaces the existing index and clears all conversation sessions. The response
contains `chunks_indexed` and `pages_processed`.

Example with curl:

```shell
curl -X POST http://localhost:8000/upload \
  -F "file=@data/sample_test_file.pdf"
```

### `POST /query`

Accepts a JSON request with a required `question`, optional `top_k` from 1 to
10 (default 3), and optional `session_id`. Omit `session_id` to start a new
conversation. Reuse the returned session ID for follow-up questions.

```json
{
  "question": "What is the list price of the Magpie-7?",
  "top_k": 3
}
```

The response contains the generated `answer`, a `sources` array, and the
`session_id`. Each source includes `chunk_id`, `text`, `page`, and `score`.

```json
{
  "answer": "The list price of the Magpie-7 is €68,400 per unit.",
  "sources": [
    {"chunk_id": 4, "text": "...", "page": 2, "score": 0.7542}
  ],
  "session_id": "returned-session-id"
}
```

### `POST /query/stream`

Accepts the same JSON request as `/query` and returns `text/event-stream`.
Each token event has a JSON `token` value. The final event contains `sources`
and `session_id`:

```text
data: {"token": "The list price"}

data: {"sources": [...], "session_id": "..."}
```

## Conversation Sessions

Sessions are stored in process memory. Each session retains the latest five
question-and-answer turns (ten messages) to support follow-up references.
Session state is lost when the server restarts, and uploading a new PDF clears
all sessions.

## Evaluation

The evaluation suite contains 19 questions across six categories, including
direct lookups, specific values, tables, multi-hop questions, negative
questions, and retrieval challenges. It queries a real LLM, so a working API
key for the selected provider and network access are required.

```shell
uv run pytest tests/test_eval.py -v
```

## Current Scope and Limitations

- The service indexes one PDF at a time; each upload replaces the previous
  document.
- Extracted text, vector and BM25 indices, and conversation sessions are held
  in memory only. There is no persistence.
- Scanned PDFs without extractable text are not processed with OCR.
- Chunks are character-based and do not carry context across page boundaries.
- Dense, fact-heavy chunks may rank lower; a larger `top_k` can help include
  them among the retrieved candidates.
- The store is a single global in-memory index and is not isolated by user.

To add a dependency, run `uv add <package>`. Run project commands through
`uv run <command>`.
