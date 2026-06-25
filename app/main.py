import uuid
import shutil
from pathlib import Path
from app.routes import login, users
from fastapi import FastAPI, UploadFile, File, Depends, HTTPException

# ===== Auth / DB imports =====
from databases.database import get_db
from databases.oauth2 import get_current_user
from databases.models import User

# ===== Schemas =====
from databases.schemas import QueryIn

# ===== Ingestion / Chunking =====
from data_preprocessing.ingest import ingest_pdf, build_documents
from data_preprocessing.chunking import split_markdown_document

# ===== Vector DB / Retrieval =====
from data_preprocessing.vector_db import (
    upsert_split_documents,
    retrieve_context,
    rerank_results
)

# ===== LLM =====
from llm.ask_llm import generation
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(tags=["Main APP"])

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],        # tighten to your frontend URL in production
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
def home():
    return {"message": "ChatPDF backend is running"}


# =========================================================
# UPLOAD ENDPOINT
# =========================================================
@app.post("/upload")
async def upload_file(
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user)
):
    """
    Upload a PDF, parse it, split into chunks, and store in vector DB.
    Returns a document_id that the frontend must send later to /generation.
    """
    try:
        if not file.filename:
            raise HTTPException(status_code=400, detail="No file name provided.")

        original_filename = file.filename
        document_id = str(uuid.uuid4())
        print("\n\n")
        print(document_id)
        print("\n\n")

        # Save file with unique internal filename
        unique_name = f"{current_user.id}_{document_id}_{original_filename}"
        saved_path = UPLOAD_DIR / unique_name

        with open(saved_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # 1) Parse PDF
        json_result = ingest_pdf(file_path=str(saved_path))

        # 2) Build LangChain Documents
        documents = build_documents(json_result, original_filename)

        # 3) Split into chunks
        nodes = split_markdown_document(documents)
        if not nodes:
            raise HTTPException(
                status_code=400,
                detail="No chunks were created from the uploaded document."
            )

        # 4) Upsert chunks into vector store
        upsert_split_documents(
            markdown_nodes=nodes,
            user_id=str(current_user.id),
            source_document=document_id
        )

        return {
            "status": "Successfully indexed and processed document",
            "document_id": document_id,
            "saved_path": str(saved_path),
            "chunks_indexed": len(nodes)
        }

    except HTTPException:
        raise
    except Exception as e:
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
            # section_title = doc.metadata.get("section_title")
            # chunk_index = doc.metadata.get("chunk_index")

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