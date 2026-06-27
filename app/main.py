import uuid
import shutil
from pathlib import Path
from app.routes import login, users
from fastapi import FastAPI, UploadFile, File, Depends, HTTPException, status

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
    get_vector_store,
    upsert_split_documents,
    retrieve_context,
    rerank_results
)
import os

# ===== LLM =====
from llm.ask_llm import generation
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
    allow_origins=["*"],       
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(login.router)
app.include_router(users.router)


# Directory to save uploaded files
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)


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
        get_vector_store()

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
    """
    Ask a question against one uploaded document.
    Frontend must send:
    {
        "query": "...",
        "document_id": "..."
    }
    """
    try:
        # 1) Retrieve chunks only from this user's selected document
        raw_results = retrieve_context(
            query=question.query,
            user_id=str(current_user.id),
            source_document=question.document_id,
        )

        if not raw_results:
            return {
                "query": question.query,
                "document_id": question.document_id,
                "answer": "I couldn't find relevant information in that document.",
                "citations": [],
                "results_count": 0
            }

        print(f"\nRetrieved {len(raw_results)} documents from vector search.")

        # 2) Rerank retrieved chunks
        final_context = rerank_results(question.query, raw_results)

        # 3) Build LLM context string with citation-friendly chunk labels
        context_blocks = []
        citations = []

        for idx, doc in enumerate(final_context):
            source_name = doc.metadata.get("source_name")
            page = doc.metadata.get("page")

            block = (
                f"[Chunk {idx + 1}]\n"
                f"Source: {source_name}\n"
                f"Page: {page}\n"
                f"Content:\n{doc.page_content}"
            )
            context_blocks.append(block)

            citations.append({
                "chunk_label": f"Chunk {idx + 1}",
                "source_name": source_name,
                "page": page,
            })

        doc_context = "\n\n---\n\n".join(context_blocks)

        # 4) Generate final answer
        answer = generation(
            query=question.query,
            doc_context=doc_context
        )
        print(f"Retrieved Context: {doc_context} \n\n Generated Answer: {answer}")
        return {
            "query": question.query,
            "document_id": question.document_id,
            "answer": answer,
            "citations": citations,
            "results_count": len(final_context)
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")