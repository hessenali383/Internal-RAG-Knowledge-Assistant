# Meridian — Internal RAG Knowledge Assistant

**Turn a company's scattered PDFs into an AI assistant that answers instantly — grounded in your own documents, with every answer cited.**

<p align="center">
  <img src="assets/screenshot-desktop.png" alt="Meridian chat interface" width="820">
</p>

---

## Overview

Every company accumulates internal knowledge — handbooks, policies, product roadmaps, compliance docs, support runbooks — scattered across dozens of PDF files. Employees waste time searching for information that already exists in an official document, ask a colleague instead of reading the manual, or worse, act on outdated information.

**Meridian** solves this with a Retrieval-Augmented Generation (RAG) pipeline: upload PDFs, they're automatically indexed into a private knowledge base, and anyone on the team can ask questions in plain language and get accurate answers **grounded in the exact document and passage they came from** — no guessing, no fabricated information.

---

## Key Features

- **Cited, grounded answers.** Every response is generated strictly from the uploaded documents. A source chip under each answer shows exactly which file and chunk it came from, so answers are always verifiable.
- **Local-first privacy.** PDF parsing, chunking, embedding, and vector storage all run locally. Only the question and the relevant retrieved snippets are sent externally — for the final answer generation step only. Raw files and the full knowledge base never leave the machine.
- **Familiar, professional chat UX.** Message bubbles, read receipts, typing/retrieval indicators — the same visual language people already know from everyday chat apps, wrapped in a distinct, brandable identity.
- **Instant knowledge base updates.** Drag a PDF into the sidebar and it's searchable within seconds — no re-deployment, no technical team required for day-to-day content updates.
- **Fully responsive.** Works as a two-pane desktop app or a single-column mobile experience with a slide-over knowledge base drawer.
- **Production-minded architecture.** Clean separation between the knowledge layer (vector store), the retrieval layer, and the generation layer (LLM), so any piece can be swapped or scaled independently.

---

## How It Works

```
 1. Upload               2. Local processing                    3. Ask & retrieve
┌──────────┐   ┌─────────────────────────────────┐   ┌────────────────────────────┐
│  *.pdf   │──▶│ Extract text → chunk → embed     │──▶│ "How many vacation days     │
└──────────┘   │ (all local, no API calls)        │   │  do full-time staff get?"   │
               │ Store in a persistent local       │   └────────────────────────────┘
               │ vector database (ChromaDB)        │                │
               └─────────────────────────────────┘                 ▼
                                                       Retrieve the most relevant
                                                       chunks → Gemini generates the
                                                       final answer, grounded in them,
                                                       with the source cited
```

---

## Getting Started

### Prerequisites

- Python 3.10+
- A [Gemini API key](https://aistudio.google.com/apikey)

### Installation

```bash
cd backend
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt

cp .env.example .env
# open .env and add your GEMINI_API_KEY

uvicorn main:app --reload
```

Open **http://localhost:8000** — FastAPI serves the chat interface directly, so nothing else needs to run.

> The first upload downloads the local embedding model (`all-MiniLM-L6-v2`, ~90MB) once. After that, embedding runs fully offline.

---

## Using Meridian

**Uploading documents**
Drag a PDF onto the sidebar dropzone, or click "Choose a PDF." A live ingestion receipt confirms how many chunks were embedded. The document then appears in the knowledge base list with its chunk count.

**Asking questions**
Type a question in plain language and press Enter (Shift+Enter for a new line). A short "Retrieving context…" indicator shows while the relevant passages are being pulled from the knowledge base, then the answer streams in with source citations underneath.

**Managing the knowledge base**
Hover any document in the sidebar to reveal a delete button, which removes all of its chunks from the vector store immediately.

**On mobile**
The knowledge base becomes a slide-over drawer, opened from the hamburger icon in the chat header, so the full experience works on a phone without any layout compromises.

---

## Customization Guide

The codebase is deliberately small and unbundled so each piece can be modified independently.

| I want to… | What to change |
|---|---|
| **Rebrand it** (name, colors, logo) | Everything lives in `frontend/index.html` — the CSS custom properties at the top of the `<style>` block (`--brand`, `--accent`, `--bg`, etc.) control the entire palette; the `Meridian` text and "M" monogram are the only hardcoded brand references. |
| **Change the language / copy** | All UI strings are plain text inside `index.html` — no i18n framework, so a straight find-and-replace (or a simple language toggle) is enough for a single-language rebrand. |
| **Swap the LLM provider** | `GeminiClient` in `rag_engine.py` is a single, self-contained class. Replace it with an equivalent client for OpenAI, Claude, or a self-hosted model — the rest of the app only calls `generate_answer()`. |
| **Swap the embedding model** | Change `EMBEDDING_MODEL_NAME` in `rag_engine.py` to any `sentence-transformers` model for better accuracy (at the cost of speed), or point it at an API-based embedding service. |
| **Swap the vector database** | The `VectorStore` class wraps ChromaDB entirely. Replacing it with Pinecone, Qdrant, or pgvector only requires reimplementing that one class — nothing else in the app is aware of the storage backend. |
| **Tune retrieval quality** | `CHUNK_SIZE`, `CHUNK_OVERLAP`, and `TOP_K` are constants at the top of `rag_engine.py`. Larger chunks give the model more context per source; a higher `TOP_K` broadens what's retrieved per question. |
| **Add authentication** | There's no auth layer in this reference build. Add it at the FastAPI level (e.g. an `Authorization` header check or a proper OAuth/session flow) before exposing this beyond a local demo. |
| **Support multiple teams/clients from one deployment** | Add a `tenant_id` field to chunk metadata in `VectorStore.add_document`, and filter every query and list call by it — the schema already supports arbitrary metadata. |
| **Stream answers instead of waiting for the full response** | Gemini's SDK supports streaming (`send_message_stream`); wire that through a `StreamingResponse` in FastAPI and update the frontend to append tokens as they arrive. |
| **Deploy it** | Containerize `backend/` with a standard Python Dockerfile, mount a persistent volume for `backend/data/`, and set `GEMINI_API_KEY` as an environment variable rather than a `.env` file in production. |

---

## Tech Stack

| Layer | Technology |
|---|---|
| Backend API | FastAPI (Python) |
| PDF parsing | pypdf |
| Chunking | Custom paragraph/sentence-aware splitter (no external dependency) |
| Embeddings | sentence-transformers, local CPU inference |
| Vector store | ChromaDB (persistent, local) |
| Answer generation | Gemini API (`google-genai`) |
| Frontend | Vanilla HTML/CSS/JS — no build step |

---

## API Reference

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/upload` | Ingest a PDF into the knowledge base |
| `GET` | `/api/documents` | List ingested documents and their chunk counts |
| `DELETE` | `/api/documents/{filename}` | Remove a document and its chunks |
| `POST` | `/api/chat` | Ask a question — body: `{ "message": "...", "history": [...] }` |
| `GET` | `/api/health` | Liveness check |

---

## Notes

This is a reference implementation intended as a strong starting point for a client engagement, not a finished multi-tenant SaaS product. The customization table above outlines the natural next steps for turning it into one.
