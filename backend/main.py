"""
main.py
-------
FastAPI application for the Meridian Internal RAG Knowledge Assistant.

Run with:
    uvicorn main:app --reload

Endpoints:
    POST   /api/upload            -> ingest a PDF into the local knowledge base
    GET    /api/documents         -> list ingested documents
    DELETE /api/documents/{name}  -> remove a document from the knowledge base
    POST   /api/chat              -> ask a question, grounded in retrieved context
    GET    /                      -> serves the chat frontend
"""

import shutil
import uuid
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv
load_dotenv()  # must run before rag_engine reads GEMINI_API_KEY

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

from rag_engine import GeminiClient, UPLOAD_DIR, VectorStore, extract_text_from_pdf

APP_NAME = "Meridian Knowledge Assistant"

app = FastAPI(title=APP_NAME)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten this to your actual frontend origin in production
    allow_methods=["*"],
    allow_headers=["*"],
)

vector_store = VectorStore()
_gemini_client: Optional[GeminiClient] = None  # created lazily so the app still boots without an API key


def get_gemini_client() -> GeminiClient:
    global _gemini_client
    if _gemini_client is None:
        _gemini_client = GeminiClient()
    return _gemini_client


# --------------------------------------------------------------------------- #
# Schemas
# --------------------------------------------------------------------------- #

class ChatTurn(BaseModel):
    role: str          # "user" | "assistant"
    content: str


class ChatRequest(BaseModel):
    message: str
    history: List[ChatTurn] = []


class ChatResponse(BaseModel):
    answer: str
    sources: List[dict]


class DocumentInfo(BaseModel):
    filename: str
    chunks: int


class UploadResponse(BaseModel):
    filename: str
    chunks_added: int
    message: str


# --------------------------------------------------------------------------- #
# Knowledge base endpoints
# --------------------------------------------------------------------------- #

@app.post("/api/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    safe_name = f"{uuid.uuid4().hex[:8]}_{file.filename}"
    dest_path = UPLOAD_DIR / safe_name

    with dest_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        text = extract_text_from_pdf(dest_path)
        chunks_added = vector_store.add_document(filename=file.filename, text=text)
    except ValueError as exc:
        dest_path.unlink(missing_ok=True)
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        dest_path.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=f"Failed to process PDF: {exc}")

    return UploadResponse(
        filename=file.filename,
        chunks_added=chunks_added,
        message=f"'{file.filename}' embedded into the knowledge base as {chunks_added} chunks.",
    )


@app.get("/api/documents", response_model=List[DocumentInfo])
async def list_documents():
    return vector_store.list_documents()


@app.delete("/api/documents/{filename}")
async def delete_document(filename: str):
    removed = vector_store.delete_document(filename)
    if removed == 0:
        raise HTTPException(status_code=404, detail="Document not found in the knowledge base.")
    return {"filename": filename, "chunks_removed": removed}


# --------------------------------------------------------------------------- #
# Chat endpoint
# --------------------------------------------------------------------------- #

@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty.")

    retrieved = vector_store.query(request.message)

    try:
        client = get_gemini_client()
        answer = client.generate_answer(
            question=request.message,
            context_chunks=retrieved,
            history=[turn.model_dump() for turn in request.history],
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Gemini API error: {exc}")

    sources = [
        {
            "filename": c["source"],
            "chunk_index": c["chunk_index"],
            "relevance": round(c["score"], 3),
            "preview": c["text"][:200].replace("\n", " ").strip() + ("…" if len(c["text"]) > 200 else ""),
        }
        for c in retrieved
    ]
    return ChatResponse(answer=answer, sources=sources)


@app.get("/api/health")
async def health():
    return {"status": "ok", "documents_indexed": not vector_store.is_empty()}


# --------------------------------------------------------------------------- #
# Serve the frontend
# --------------------------------------------------------------------------- #

FRONTEND_DIR = Path(__file__).parent.parent / "frontend"

if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

    @app.get("/")
    async def serve_frontend():
        return FileResponse(str(FRONTEND_DIR / "index.html"))
