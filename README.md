# Meridian — Internal RAG Knowledge Assistant

A self-hosted Retrieval-Augmented Generation assistant: upload internal PDFs,
they're chunked and embedded into a local vector store, and a chat interface
answers questions grounded strictly in that content — citing the source
document for every answer.

Everything runs locally except the final answer-generation call, which uses
the Gemini API.

## How it works

```
 PDF upload            Local processing                     Chat
┌──────────┐   ┌──────────────────────────────┐   ┌──────────────────────┐
│  *.pdf   │──▶│ 1. Extract text (pypdf)      │   │ User question        │
└──────────┘   │ 2. Chunk (custom splitter)   │   │   │                   │
               │ 3. Embed (sentence-          │   │   ▼                   │
               │    transformers, local CPU)  │   │ Embed question        │
               │ 4. Store (ChromaDB,          │──▶│   │                   │
               │    persisted on disk)        │   │   ▼                   │
               └──────────────────────────────┘   │ Retrieve top-k chunks │
                                                    │   │                   │
                                                    │   ▼                   │
                                                    │ Gemini API answers,   │
                                                    │ grounded in context   │
                                                    └──────────────────────┘
```

Only the last step — generating the final answer — leaves the machine, and
only the retrieved text snippets and the question are sent, never the raw
files or the full vector database.

## Project structure

```
meridian-rag-assistant/
├── backend/
│   ├── main.py            FastAPI app — routes + serves the frontend
│   ├── rag_engine.py       PDF parsing, chunking, vector store, Gemini client
│   ├── requirements.txt
│   ├── .env.example
│   └── data/               local PDF + vector store storage (git-ignored)
├── frontend/
│   └── index.html          Chat UI (vanilla HTML/CSS/JS, no build step)
└── README.md
```

## Setup

**Requirements:** Python 3.10+, a [Gemini API key](https://aistudio.google.com/apikey).

```bash
cd backend
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt

cp .env.example .env
# edit .env and add your GEMINI_API_KEY

uvicorn main:app --reload
```

Open **http://localhost:8000** — the FastAPI server serves the chat interface
directly, so there's nothing else to run.

The first upload will download the local embedding model
(`all-MiniLM-L6-v2`, ~90MB) the first time it's used; after that it runs
fully offline.

## Configuration

All configuration lives in `backend/.env`:

| Variable         | Required | Description                                  |
|------------------|----------|-----------------------------------------------|
| `GEMINI_API_KEY` | Yes      | Your Gemini API key                          |
| `GEMINI_MODEL`   | No       | Defaults to `gemini-3.6-flash`                |

Chunking and retrieval parameters (`CHUNK_SIZE`, `CHUNK_OVERLAP`, `TOP_K`) are
constants at the top of `rag_engine.py`.

