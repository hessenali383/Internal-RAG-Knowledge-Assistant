"""
rag_engine.py
-------------
Core Retrieval-Augmented Generation engine for the Meridian Knowledge Assistant.

Pipeline:
  1. Extract text from an uploaded PDF (pypdf)
  2. Split it into overlapping, paragraph-aware chunks
  3. Embed chunks locally (sentence-transformers) and persist them in a
     local ChromaDB collection — no data ever leaves the machine at this stage
  4. At query time, embed the user's question, retrieve the most relevant
     chunks, and ask Gemini to answer using only that retrieved context

Everything except the final answer-generation call runs entirely locally.
"""

from __future__ import annotations

import os
import re
import uuid
from pathlib import Path
from typing import Any, Dict, List

import chromadb
from chromadb.utils import embedding_functions
from pypdf import PdfReader
from google import genai
from google.genai import types as genai_types

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
VECTOR_DB_DIR = DATA_DIR / "vector_store"
UPLOAD_DIR = DATA_DIR / "uploads"

COLLECTION_NAME = "knowledge_base"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"   # local CPU model, 384 dimensions
GEMINI_MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-3.6-flash")

CHUNK_SIZE = 900       # target characters per chunk
CHUNK_OVERLAP = 150    # characters of overlap carried into the next chunk
TOP_K = 5              # chunks retrieved per question

for _dir in (DATA_DIR, VECTOR_DB_DIR, UPLOAD_DIR):
    _dir.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------- #
# 1. PDF extraction
# --------------------------------------------------------------------------- #

def extract_text_from_pdf(file_path: Path) -> str:
    """Extract text from every page of a PDF, tagging page numbers."""
    reader = PdfReader(str(file_path))
    pages: List[str] = []
    for i, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:
            pages.append(f"[[page {i}]]\n{text}")
    if not pages:
        raise ValueError("No extractable text found in this PDF (it may be a scanned image).")
    return "\n\n".join(pages)


# --------------------------------------------------------------------------- #
# 2. Chunking — paragraph/sentence-aware, dependency-free
# --------------------------------------------------------------------------- #

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """
    Break text into overlapping chunks without cutting mid-sentence where
    avoidable. Falls back to sentence, then hard, splitting for long runs.
    """
    text = re.sub(r"\n{3,}", "\n\n", text.strip())
    if not text:
        return []

    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: List[str] = []
    buffer = ""

    def push_buffer() -> str:
        """Flush buffer to chunks list, return the overlap tail to seed the next one."""
        nonlocal buffer
        cleaned = buffer.strip()
        if cleaned:
            chunks.append(cleaned)
        tail = cleaned[-overlap:] if overlap and cleaned else ""
        buffer = ""
        return tail

    for para in paragraphs:
        candidate = f"{buffer}\n\n{para}" if buffer else para

        if len(candidate) <= chunk_size:
            buffer = candidate
            continue

        # Paragraph doesn't fit — flush what we have, then handle the paragraph
        tail = push_buffer()
        buffer = tail

        if len(para) <= chunk_size:
            buffer = f"{buffer}\n\n{para}" if buffer else para
            continue

        # Paragraph itself exceeds chunk_size — split on sentence boundaries
        sentences = re.split(r"(?<=[.!?])\s+", para)
        for sent in sentences:
            candidate = f"{buffer} {sent}".strip() if buffer else sent
            if len(candidate) <= chunk_size:
                buffer = candidate
            else:
                tail = push_buffer()
                buffer = f"{tail} {sent}".strip() if tail else sent
                # Extremely long single sentence: hard-split as a last resort
                while len(buffer) > chunk_size:
                    chunks.append(buffer[:chunk_size])
                    buffer = buffer[chunk_size - overlap:]

    push_buffer()
    return chunks


# --------------------------------------------------------------------------- #
# 3. Local vector store (ChromaDB + sentence-transformers)
# --------------------------------------------------------------------------- #

class VectorStore:
    def __init__(self) -> None:
        self._client = chromadb.PersistentClient(path=str(VECTOR_DB_DIR))
        self._embedder = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=EMBEDDING_MODEL_NAME
        )
        self._collection = self._client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=self._embedder,
            metadata={"hnsw:space": "cosine"},
        )

    def add_document(self, filename: str, text: str) -> int:
        """Chunk a document's text and upsert it into the vector store."""
        chunks = chunk_text(text)
        if not chunks:
            raise ValueError("Document produced no usable chunks.")

        ids = [f"{filename}::{i}::{uuid.uuid4().hex[:8]}" for i in range(len(chunks))]
        metadatas = [{"source": filename, "chunk_index": i} for i in range(len(chunks))]

        self._collection.add(ids=ids, documents=chunks, metadatas=metadatas)
        return len(chunks)

    def query(self, question: str, top_k: int = TOP_K) -> List[Dict[str, Any]]:
        if self._collection.count() == 0:
            return []
        results = self._collection.query(query_texts=[question], n_results=min(top_k, self._collection.count()))
        matches: List[Dict[str, Any]] = []
        for doc, meta, dist in zip(
            results["documents"][0], results["metadatas"][0], results["distances"][0]
        ):
            matches.append({"text": doc, "source": meta.get("source"), "chunk_index": meta.get("chunk_index"), "score": 1 - dist})
        return matches

    def list_documents(self) -> List[Dict[str, Any]]:
        data = self._collection.get()
        counts: Dict[str, int] = {}
        for meta in data.get("metadatas", []):
            src = meta.get("source", "unknown")
            counts[src] = counts.get(src, 0) + 1
        return [{"filename": name, "chunks": n} for name, n in sorted(counts.items())]

    def delete_document(self, filename: str) -> int:
        existing = self._collection.get(where={"source": filename})
        ids = existing.get("ids", [])
        if ids:
            self._collection.delete(ids=ids)
        return len(ids)

    def is_empty(self) -> bool:
        return self._collection.count() == 0


# --------------------------------------------------------------------------- #
# 4. Answer generation (Gemini)
# --------------------------------------------------------------------------- #

SYSTEM_INSTRUCTION = (
    "You are Meridian, an internal knowledge assistant. Answer the user's "
    "question using ONLY the context excerpts provided below, which come "
    "from the organization's own documents. "
    "If the context does not contain the answer, say so plainly instead of "
    "guessing. Be concise, professional, and cite the source file name(s) "
    "you drew from at the end of your answer in the form (source: filename.pdf)."
)


class GeminiClient:
    def __init__(self) -> None:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GEMINI_API_KEY is not set. Add it to your .env file before starting the server."
            )
        self._client = genai.Client(api_key=api_key)

    def generate_answer(
        self, question: str, context_chunks: List[Dict[str, Any]], history: List[Dict[str, str]] | None = None
    ) -> str:
        if context_chunks:
            context_block = "\n\n---\n\n".join(
                f"Source: {c['source']} (chunk {c['chunk_index']})\n{c['text']}" for c in context_chunks
            )
        else:
            context_block = "(No documents have been uploaded to the knowledge base yet.)"

        prompt = (
            f"Context excerpts:\n{context_block}\n\n"
            f"User question: {question}"
        )

        chat = self._client.chats.create(
            model=GEMINI_MODEL_NAME,
            config=genai_types.GenerateContentConfig(system_instruction=SYSTEM_INSTRUCTION),
            history=_to_gemini_history(history or []),
        )
        response = chat.send_message(prompt)
        return response.text


def _to_gemini_history(history: List[Dict[str, str]]) -> List[genai_types.Content]:
    """Convert simple {role, content} turns into the SDK's Content objects."""
    formatted: List[genai_types.Content] = []
    for turn in history:
        role = "model" if turn.get("role") == "assistant" else "user"
        formatted.append(
            genai_types.Content(role=role, parts=[genai_types.Part.from_text(text=turn.get("content", ""))])
        )
    return formatted
