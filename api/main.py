"""
api/main.py — FastAPI entry point.

Endpoints:
  GET    /                    health check
  POST   /upload/             ingest a PDF
  GET    /documents/          list all ingested documents
  GET    /documents/{doc_id}  get a single document record
  DELETE /documents/{doc_id}  remove a document record
  POST   /ask/                ask a question (with optional conversation history)
"""

import shutil
import uuid
import logging
from pathlib import Path
from typing import Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, UploadFile, File, status
from pydantic import BaseModel

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy singleton state — populated in lifespan, not at import time
# ---------------------------------------------------------------------------

_state: dict = {}


def get_vector_db():
    db = _state.get("vector_db")
    if db is None:
        raise HTTPException(status_code=503, detail="Vector DB not ready")
    return db


def get_llm():
    llm = _state.get("llm")
    if llm is None:
        raise HTTPException(status_code=503, detail="LLM not ready")
    return llm


def get_db():
    db = _state.get("db")
    if db is None:
        raise HTTPException(status_code=503, detail="Database not ready")
    return db


# ---------------------------------------------------------------------------
# Lifespan: initialise singletons AFTER uvicorn is up, not at import time
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Start-up: load models and connect to services. Shut-down: nothing to do."""
    from databases.database import Database
    from data_preprocessing.vector_db import VectorDB
    from llm.llm_connection import LLM

    logger.info("🚀 Starting up — loading models and connecting to services...")

    try:
        _state["db"] = Database()
        logger.info("✅ PostgreSQL connected")
    except Exception as e:
        logger.error(f"❌ Database init failed: {e}")
        # App still starts — DB endpoints will 503 but /upload and /ask won't crash

    try:
        _state["vector_db"] = VectorDB()
        logger.info("✅ Qdrant connected")
    except Exception as e:
        logger.error(f"❌ VectorDB init failed: {e}")

    try:
        _state["llm"] = LLM()
        logger.info("✅ LLM ready")
    except Exception as e:
        logger.error(f"❌ LLM init failed: {e}")

    yield  # app is now serving requests

    logger.info("👋 Shutting down")
    _state.clear()


app = FastAPI(title="ChatPDF API", lifespan=lifespan)

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "document_files"
DATA_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class AskRequest(BaseModel):
    question: str
    source: Optional[str] = None           # filter results to a specific PDF
    conversation_id: Optional[str] = None  # enables multi-turn history


class AskResponse(BaseModel):
    answer: str
    conversation_id: str
    sources: list


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/")
def homepage():
    return {"status": "ok", "message": "ChatPDF API is running"}


@app.post("/upload/", status_code=status.HTTP_201_CREATED)
async def upload_file(file: UploadFile = File(...)):
    """Accept a PDF, ingest it into the vector store, and record it in the DB."""
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only PDF files are allowed.",
        )

    from data_preprocessing.ingest import load_pdf
    from data_preprocessing.chunking import chunk_pdf

    file_path = DATA_DIR / file.filename
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        pages = load_pdf(file.filename)
        child_chunks, _ = chunk_pdf(pages)

        if not child_chunks:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="PDF contained no extractable text. Check if it is image-only.",
            )

        get_vector_db().build_index(child_chunks, source_name=file.filename)
        db_result = get_db().add_document(source=file.filename)

    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Upload failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        )

    return {
        "file": file.filename,
        "chunks_indexed": len(child_chunks),
        "db_result": db_result,
    }


@app.get("/documents/")
def list_documents():
    return get_db().list_documents()


@app.get("/documents/{doc_id}")
def get_document(doc_id: int):
    doc = get_db().get_document(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    return doc


@app.delete("/documents/{doc_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_document(doc_id: int):
    if not get_db().delete_document(doc_id):
        raise HTTPException(status_code=404, detail="Document not found")


@app.post("/ask/", response_model=AskResponse)
def ask_question(body: AskRequest):
    """Answer a question using hybrid RAG with optional conversation history."""
    conv_id = body.conversation_id or str(uuid.uuid4())
    history = get_db().get_conversation(conv_id)

    context = get_vector_db().search(
        query=body.question,
        top_k=5,
        source=body.source,
    )

    if not context:
        # Return a graceful answer rather than a hard 404
        return AskResponse(
            answer="The answer to your question is not found in the uploaded document.",
            conversation_id=conv_id,
            sources=[],
        )

    answer = get_llm().generate_response(
        query=body.question,
        context=context,
        history=history,
    )

    get_db().add_message(conv_id, role="user", content=body.question)
    get_db().add_message(conv_id, role="assistant", content=answer)

    return AskResponse(
        answer=answer,
        conversation_id=conv_id,
        sources=[{"source": c["source"], "page": c["page"]} for c in context],
    )
