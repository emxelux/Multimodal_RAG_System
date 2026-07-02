import uuid
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from app.routes import login, users
from fastapi import FastAPI, UploadFile, File, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
import json

from dotenv import load_dotenv
# ===== Auth / DB imports =====
from databases.database import get_db
from sqlalchemy.orm import Session
from databases.utils import hash_pdf
from databases.oauth2 import get_current_user
from databases.models import User, Document

# ===== Schemas =====
from databases.schemas import QueryIn, DocumentIn

# ===== Ingestion / Chunking =====
from data_preprocessing.ingest import ingest_pdf, build_documents
from data_preprocessing.chunking import split_markdown_document

# ===== Vector DB / Retrieval =====
from data_preprocessing.vector_db import (
    upsert_split_documents,
    retrieve_context,
    rerank_results
)
import os

# ===== LLM =====
from llm.ask_llm import generation, stream_generation
from fastapi.middleware.cors import CORSMiddleware

from databases.database import engine
from databases import models  # Make sure this imports the file where your "User" model lives

# This line tells SQLAlchemy to physically create tables in Postgres if they don't exist
models.Base.metadata.create_all(bind=engine)

load_dotenv()

app = FastAPI(tags=["Main APP"])

os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")


app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://chatmypdf.netlify.app"],       
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"]
)

app.include_router(login.router)
app.include_router(users.router)


# Directory to save uploaded files
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

CHAT_HISTORY_STORE: Dict[Tuple[str, str], List[Dict[str, str]]] = {}


# =========================================================
# SAFE FIELD ACCESSOR
# =========================================================
def _field(r: Any, key: str, default: Any = None) -> Any:
    """
    Safely pull a field off a retrieval result, regardless of whether
    `r` is a plain dict, a LangChain-style Document (page_content +
    metadata), or some other Pydantic/object model.

    This fixes: AttributeError: 'Document' object has no attribute 'get'
    which happened because retrieve_context()/rerank_results() return
    Document objects, not dicts, but the endpoint assumed dicts.
    """
    # 1) Plain dict
    if isinstance(r, dict):
        return r.get(key, default)

    # 2) LangChain-style Document: metadata dict + page_content
    metadata = getattr(r, "metadata", None)
    if key == "content":
        page_content = getattr(r, "page_content", None)
        if page_content is not None:
            return page_content
    if isinstance(metadata, dict) and key in metadata:
        return metadata[key]

    # 3) Direct attribute on the object itself (custom Pydantic model)
    if hasattr(r, key):
        return getattr(r, key)

    # 4) Pydantic model_dump() fallback (covers nested/aliased fields)
    if hasattr(r, "model_dump"):
        try:
            return r.model_dump().get(key, default)
        except Exception:
            pass

    return default


@app.get("/")
def health_check():
    return {"message": "ChatPDF backend is running"}


# =========================================================
# UPLOAD ENDPOINT
# =========================================================
@app.post("/upload")
async def upload_file(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Upload a PDF, parse it, split into chunks, and store in vector DB.
    Checks file content hash to prevent duplicate parsing and storage overhead.
    Returns a document_id that the frontend must send later to /generation.
    """
    try:
        if not file.filename:
            raise HTTPException(status_code=400, detail="No file name provided.")

        original_filename = file.filename
        document_id = str(uuid.uuid4())

        # 1) Temporary save to run hash_pdf on the physical file
        unique_name = f"{current_user.id}_{document_id}_{original_filename}"
        saved_path = UPLOAD_DIR / unique_name

        with open(saved_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # 2) Compute hash and check for duplicates in the DB
        hashed_content = hash_pdf(saved_path)
        existing_file = db.query(Document).filter(Document.document_hash == hashed_content).first()
        
        if existing_file:
            # Clean up the file we just saved to avoid redundant disk usage
            if saved_path.exists():
                saved_path.unlink()
                
            return {
                "status": "Document already exists and is indexed",
                "document_id": existing_file.id,
                "chunks_indexed": getattr(existing_file, "chunk_count", 0),  # Falls back safely if not explicitly in your schema
                "duplicated": True
            }

        # 3) Process new document if no duplicate is found
        # Parse PDF
        json_result = ingest_pdf(file_path=str(saved_path))

        # Build LangChain Documents
        documents = build_documents(json_result, original_filename)
        # Split into chunks
        nodes = split_markdown_document(documents)
        if not nodes:
            if saved_path.exists():
                saved_path.unlink()
            raise HTTPException(
                status_code=400,
                detail="No chunks were created from the uploaded document."
            )

        # Upsert chunks into vector store
        upsert_split_documents(
            markdown_nodes=nodes,
            user_id=str(current_user.id),
            source_document=document_id
        )

        # 4) Save metadata & hash record to relational DB
        new_doc = Document(
            id=document_id,
            document_hash=hashed_content,
            user_id=current_user.id
        )
        # Handle chunk_count dynamically if it exists on your Document model
        if hasattr(new_doc, 'chunk_count'):
            new_doc.chunk_count = len(nodes)
            
        db.add(new_doc)
        db.commit()

        return {
            "status": "Successfully indexed and processed document",
            "document_id": document_id,
            "chunks_indexed": len(nodes),
            "duplicated": False
        }

    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


# =========================================================
# GENERATION ENDPOINT
# =========================================================
@app.post("/generation")
def retrieval_and_generation(
    question: QueryIn,
    current_user: User = Depends(get_current_user)
):
    # ── Step 1: retrieval + reranking (synchronous, happens before streaming) ──
    try:
        results = retrieve_context(question.query, current_user.id, question.document_id)
        final_context = rerank_results(question.query, results)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Retrieval failed: {str(e)}")

    doc_context = "\n\n".join([
        f"[Chunk {i+1}] Source: {_field(r, 'source_name', '')}, Page: {_field(r, 'page', '?')}\n{_field(r, 'content', '')}"
        for i, r in enumerate(final_context)
    ])

    citations = [
        {
            "chunk_label": f"Chunk {i+1}",
            "source_name": _field(r, "source_name", ""),
            "page": _field(r, "page")
        }
        for i, r in enumerate(final_context)
    ]

    history_key = (str(current_user.id), question.document_id)
    prior_history = question.history or CHAT_HISTORY_STORE.get(history_key, [])

    def event_stream():
        yield f"data: {json.dumps({'type': 'citations', 'citations': citations, 'results_count': len(final_context)})}\n\n"

        response_parts: List[str] = []
        for token in stream_generation(
            query=question.query,
            doc_context=doc_context,
            history=prior_history,
        ):
            response_parts.append(token)
            yield f"data: {json.dumps({'type': 'token', 'token': token})}\n\n"

        assistant_reply = "".join(response_parts)
        updated_history = list(prior_history)
        updated_history.append({"role": "user", "content": question.query})
        updated_history.append({"role": "assistant", "content": assistant_reply})
        CHAT_HISTORY_STORE[history_key] = updated_history

        yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        }
    )