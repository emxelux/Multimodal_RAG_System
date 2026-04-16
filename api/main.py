"""
api/main.py — FastAPI entry point.

Endpoints:
  GET  /                      health check
  POST /upload/               ingest a PDF
  GET  /documents/            list all ingested documents
  GET  /documents/{doc_id}    get a single document record
  DELETE /documents/{doc_id}  remove a document record
  POST /ask/                  ask a question (with optional conversation history)
"""

import shutil
import uuid
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, UploadFile, File, status
from pydantic import BaseModel

from databases.database import Database
from data_preprocessing.vector_db import VectorDB
from data_preprocessing.ingest import load_pdf, DATA_DIR
from data_preprocessing.chunking import chunk_pdf
from data_preprocessing.embedding import dense_embeddings
from llm.llm_connection import LLM

app = FastAPI(title="ChatPDF API")

DATA_DIR.mkdir(exist_ok=True)

# Shared singletons (initialised once at startup)
my_client = VectorDB()
llm = LLM()
db = Database()


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class AskRequest(BaseModel):
    question: str
    source: Optional[str] = None          # filter results to a specific PDF
    conversation_id: Optional[str] = None  # pass to enable history


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

    file_path = DATA_DIR / file.filename
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        pages = load_pdf(file.filename)
        child_chunks, _parent_docs = chunk_pdf(pages, dense_embeddings)
        my_client.build_index(child_chunks, source_name=file.filename)
        result = db.add_document(source=file.filename)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        )

    return {
        "file": file.filename,
        "chunks_indexed": len(child_chunks),
        "db_result": result,
    }


@app.get("/documents/")
def list_documents():
    """Return all documents that have been ingested."""
    return db.list_documents()


@app.get("/documents/{doc_id}")
def get_document(doc_id: int):
    doc = db.get_document(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    return doc


@app.delete("/documents/{doc_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_document(doc_id: int):
    deleted = db.delete_document(doc_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Document not found")


@app.post("/ask/", response_model=AskResponse)
def ask_question(body: AskRequest):
    """
    Answer a question using RAG.
    Pass conversation_id on follow-up turns to include prior context.
    """
    conv_id = body.conversation_id or str(uuid.uuid4())

    # Retrieve conversation history (last 20 messages)
    history = db.get_conversation(conv_id)

    # Hybrid vector search
    context = my_client.search(
        query=body.question,
        top_k=5,
        source=body.source,
    )

    if not context:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No relevant context found. Make sure a document has been uploaded.",
        )

    # Generate answer
    answer = llm.generate_response(
        query=body.question,
        context=context,
        history=history,
    )

    # Persist this turn
    db.add_message(conv_id, role="user", content=body.question)
    db.add_message(conv_id, role="assistant", content=answer)

    return AskResponse(
        answer=answer,
        conversation_id=conv_id,
        sources=[
            {"source": c["source"], "page": c["page"]}
            for c in context
        ],
    )
