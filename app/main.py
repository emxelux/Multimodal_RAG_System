import uuid
import shutil
from pathlib import Path
from typing import Any
from app.routes import login, users
from fastapi import FastAPI, UploadFile, File, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
import json
import logging
from dotenv import load_dotenv
import sys


# # ===== Schemas =====
from databases.schemas import QueryIn, DocumentIn


import os

# # ===== LLM =====
from fastapi.middleware.cors import CORSMiddleware


from app.logger_config import setup_logging


# 1. Initialize configuration ONCE
setup_logging()

# 2. Grab module logger
logger = logging.getLogger(__name__)


logger.info("================== APP IS STARTING UP ====================")


from databases.database import engine
from databases import models  

models.Base.metadata.create_all(bind=engine)

load_dotenv()



app = FastAPI(tags=["Main APP"])

os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")

@app.on_event("startup")
async def startup_event():
    log_memory("APPLICATION STARTUP")

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

# # =========================================================
# # SAFE FIELD ACCESSOR
# # =========================================================
def _field(r: Any, key: str, default: Any = None) -> Any:
    
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


from databases.database import get_db
from sqlalchemy.orm import Session
from databases.oauth2 import get_current_user
from databases.models import User, Document
# @app.post("/upload")
# async def upload_file(
#     file: UploadFile = File(...),
#     db: Session = Depends(get_db),
#     current_user: User = Depends(get_current_user)
# ):
#     """
#     Upload a PDF, parse it, split into chunks, and store in vector DB.
#     Checks file content hash to prevent duplicate parsing and storage overhead.
#     Returns a document_id that the frontend must send later to /generation.
#     """
#     from databases.utils import hash_pdf
#     from data_preprocessing.ingest import ingest_pdf, build_documents
#     from data_preprocessing.chunking import split_markdown_document


#     from data_preprocessing.vector_db import (
#     upsert_split_documents,
#     retrieve_context,
#     rerank_results
# )
#     try:
#         logger.info('=========== ✔️ Starting ingestion and uploading ✔️ ====================')
#         if not file.filename:
#             logger.error("XXXXXXXXXXXXXXXXXXX   No File name Provided XXXXXXXXXXXXXXXXXXXXXX")
#             raise HTTPException(status_code=400, detail="No file name provided.")

#         original_filename = file.filename
#         document_id = str(uuid.uuid4())
#         logger.info(F" ==================== INGESTING {document_id} ============================")

       
#         unique_name = f"{current_user.id}_{document_id}_{original_filename}"
#         saved_path = UPLOAD_DIR / unique_name

#         with open(saved_path, "wb") as buffer:
#             shutil.copyfileobj(file.file, buffer)

#         # 2) Compute hash and check for duplicates in the DB
#         logger.info("==================== STARTING HASHING PDF =======================")
#         hashed_content = hash_pdf(saved_path)
#         existing_file = db.query(Document).filter(Document.document_hash == hashed_content).first()
        
#         if existing_file:
#             logger.warning("================================ PDF ALREADY EXISTs ===============================")
#             # Clean up the file we just saved to avoid redundant disk usage
#             if saved_path.exists():
#                 saved_path.unlink()
                
#             return {
#                 "status": "Document already exists and is indexed",
#                 "document_id": existing_file.id,
#                 "chunks_indexed": getattr(existing_file, "chunk_count", 0),  # Falls back safely if not explicitly in your schema
#                 "duplicated": True
#             }

#         # 3) Process new document if no duplicate is found
#         # Parse PDF
#         json_result = ingest_pdf(file_path=str(saved_path))
#         if not json_result:
#             logger.error(" ======================== THERE IS NO JSON RESULT CREATED =============================")
            
        

#         # Build LangChain Documents
#         documents = build_documents(json_result, original_filename)
#         if not documents:
#             logger.error("XXXXXXXXXXXXXXXXXXX  COULD NOT SUCCESSFULLY CREATE DOCUMENT OBJECT  XXXXXXXXXXXXXXXXXXXXXX")
#         logger.info("=========================  DOCUMENT OBJECT BUILT SUCCESSFULLY ===========================")
#         # Split into chunks
#         nodes = split_markdown_document(documents)
#         if not nodes:
#             logger.error("==========================   NO CHUNKS WERE CREATED SUCCESSFULLY  ================================")
#             if saved_path.exists():
#                 saved_path.unlink()
#             raise HTTPException(
#                 status_code=400,
#                 detail="No chunks were created from the uploaded document."
#             )
#         logger.info(f" =====================  {len(nodes)} NODES CREATED SUCCESSFULY  ==========================================")
#         # Upsert chunks into vector store
#         upsert_split_documents(
#             markdown_nodes=nodes,
#             user_id=str(current_user.id),
#             source_document=document_id
#         )
#         logger.info("DOCUMENT UPSERTED TO DATABASE SUCCESSFULLY")
#         # 4) Save metadata & hash record to relational DB
#         new_doc = Document(
#             id=document_id,
#             document_hash=hashed_content,
#             user_id=current_user.id
#         )
        
#         # Handle chunk_count dynamically if it exists on your Document model
#         if hasattr(new_doc, 'chunk_count'):
#             new_doc.chunk_count = len(nodes)
            
#         db.add(new_doc)
#         db.commit()
#         logger.info(" ============================   NEW DOCUMENT ADDED TO DATABASE SUCCESSFULLY  =============================")

#         return {
#             "status": "Successfully indexed and processed document",
#             "document_id": document_id,
#             "chunks_indexed": len(nodes),
#             "duplicated": False
#         }

    
#     except Exception as e:
#         db.rollback()
#         logger.error(f"THEREIS AN ERROR {e}")
#         raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")




import logging
from fastapi import BackgroundTasks
from databases.database import SessionLocal

logger = logging.getLogger(__name__)

# ── background worker ──────────────────────────────────────────
def process_document_upload(
    saved_path: Path,
    document_id: str,
    user_id: str,
    hashed_content: str,
    original_filename: str,
):
    
    from data_preprocessing.ingest import ingest_pdf, build_documents
    from data_preprocessing.chunking import split_markdown_document
    from data_preprocessing.vector_db import upsert_split_documents

    db = SessionLocal()
    doc_row = db.query(Document).filter(Document.id == document_id).first()

    try:
        logger.info(f"[{document_id}] starting ingestion")
        # log_memory("----------------------- BEFORE STARTING INGESTION -------------------------------")
        json_result = ingest_pdf(file_path=str(saved_path))
        if not json_result:
            raise ValueError("Parser returned no content")

        # log_memory("---------------------------- BEFORE BUILDING DOCUMENTS ---------------------------")
        documents = build_documents(json_result, original_filename)
        # log_memory("---------------------------- AFTER BUILDING DOCUMENTS ---------------------------")
        if not documents:
            raise ValueError("Could not build document objects from parsed content")
        # log_memory("---------------------- BEFORE SPLITTING DOCUMENTS -----------------------")
        nodes = split_markdown_document(documents)
        # log_memory("------------------------ AFTER SPLITTING DOCUMENTS -------------------------")
        if not nodes:
            raise ValueError("No chunks were created from the uploaded document")


        # log_memory(" ------------------------- BEFORE UPSERTING TO VECTORDB -------------------------")
        upsert_split_documents(
            markdown_nodes=nodes,
            user_id=str(user_id),
            source_document=document_id,
        )
        log_memory("---------------------------- AFTER UPSERTING TO VECTORDB --------------------------------")

        doc_row.status = "completed"
        doc_row.chunk_count = len(nodes)
        db.commit()
        logger.info(f"[{document_id}] completed — {len(nodes)} chunks")

    except Exception as e:
        db.rollback()
        doc_row = db.query(Document).filter(Document.id == document_id).first()
        doc_row.status = "failed"
        doc_row.error_message = str(e)[:500]
        db.commit()
        logger.error(f"[{document_id}] failed: {e}")

    finally:
        db.close()
        if saved_path.exists():
            saved_path.unlink()


import os
import psutil

process = psutil.Process(os.getpid())

def log_memory(stage: str):
    memory_mb = process.memory_info().rss / (1024 * 1024)
    logger.info(f"[MEMORY] {stage}: {memory_mb:.2f} MB")

# ── upload endpoint: now returns immediately ───────────────────
@app.post("/upload")
async def upload_file(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    logger.info("============= GETTING INGESTION AND PROCESSING STARTED ======================")
    from databases.utils import hash_pdf
    # log_memory("--------------------------------------- DURING FILE UPLOAD ------------------------")
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file name provided.")

    original_filename = file.filename
    document_id = str(uuid.uuid4())
    unique_name = f"{current_user.id}_{document_id}_{original_filename}"
    saved_path = UPLOAD_DIR / unique_name

    with open(saved_path, "wb") as buffer:
        # log_memory("--------------------------------------- DURING FILE UPLOAD ------------------------")
        shutil.copyfileobj(file.file, buffer)

    hashed_content = hash_pdf(saved_path)
    # log_memory("------------------------ DURING CONTENT HASHING -------------------------------------------")
    existing_file = db.query(Document).filter(Document.document_hash == hashed_content).first()

    if existing_file:
        # log_memory("----------------------------------------------- DURING QUERYING DATABASE FOR DUPLICATE DOCUMENT  ----------------------------")
        logger.warning("=============================== UPLOADED FILE ALREADY EXISTS ==========================")
        saved_path.unlink()
        return {
            "status": existing_file.status,
            "document_id": existing_file.id,
            "chunks_indexed": existing_file.chunk_count or 0,
            "duplicated": True,
        }

    new_doc = Document(
        id=document_id,
        document_hash=hashed_content,
        user_id=current_user.id,
        status="processing",
    )
    db.add(new_doc)
    db.commit()
    logger.info("============================ ADDED DOCUMENT TO DATABASE SUCCESSFULLY ======================================")

    background_tasks.add_task(
        process_document_upload,
        saved_path,
        document_id,
        current_user.id,
        hashed_content,
        original_filename,
    )

    return {
        "status": "processing",
        "document_id": document_id,
        "duplicated": False,
    }


# ── new: poll this from the frontend ───────────────────────────
@app.get("/documents/{document_id}/status")
def get_document_status(
    document_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    doc = (
        db.query(Document)
        .filter(Document.id == document_id, Document.user_id == current_user.id)
        .first()
    )
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    return {
        "document_id": doc.id,
        "status": doc.status,
        "chunks_indexed": doc.chunk_count,
        "error": doc.error_message,
    }


# # =========================================================
# # GENERATION ENDPOINT
# # =========================================================
@app.post("/generation")
def retrieval_and_generation(
    question: QueryIn,
    current_user: User = Depends(get_current_user)
):
    
    from llm.ask_llm import generation, stream_generation

    from data_preprocessing.vector_db import (
    retrieve_context,
    rerank_results
)
  
    # ── Step 1: retrieval + reranking (synchronous, happens before streaming) ──
    try:
        logger.info("QUERYING THE VECTOR DATABASE....")
        results = retrieve_context(question.query, current_user.id, question.document_id)
        logger.info(f"/n/n=============/n RESULTS: {results}/n/n ==================")
        final_context = rerank_results(question.query, results)

        logger.info(f"====================== Reranked Results =====================\n\n {final_context}")
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

    def event_stream():
        yield f"data: {json.dumps({'type': 'citations', 'citations': citations, 'results_count': len(final_context)})}\n\n"

        for token in stream_generation(
            query=question.query,
            doc_context=doc_context,
        ):
            yield f"data: {json.dumps({'type': 'token', 'token': token})}\n\n"

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