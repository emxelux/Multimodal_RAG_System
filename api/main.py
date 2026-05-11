import os
import uuid
from databases.database import Database, Document
from pathlib import Path
from data_preprocessing.ingest import load_document, create_nodes
from data_preprocessing.chunking import chunk_nodes
from data_preprocessing.embedding import dense_embedding, sparse_embedding
from data_preprocessing.vector_db import RAGVectorStore
from llm.llm_connection import LLM
from databases.database import Database
from fastapi import FastAPI, HTTPException, UploadFile, File, status

app = FastAPI()
UPLOAD_DIR = "document_files"
vec_db = RAGVectorStore(dense_embedding = dense_embedding, sparse_embedding = sparse_embedding)

active_document = None
db = Database()
llm = LLM()

@app.get("/")
def check_page_status():
    return {"response":"Hello! The app is working Perfectly"}

@app.post("/upload_pdf")
async def upload_file(file: UploadFile = File(...)):
    global active_document

    file_path = os.path.join(UPLOAD_DIR, file.filename)

    with open(file_path, "wb") as f:
        content = await file.read()
        f.write(content)

    docs = load_document(file.filename)
    parent_nodes = create_nodes(docs)
    for node in parent_nodes:
        parent_id = str(uuid.uuid4())
        node.metadata["parent_id"] = parent_id

        db.add_document(
            source=file.filename,
            parent_id=parent_id,
            parent_metadata=node.metadata,
            parent_content=node.get_content(),
        )

    child_nodes = chunk_nodes(parent_nodes)
    for child in child_nodes:
        if "parent_id" not in child.metadata:
            child.metadata["parent_id"] = child.metadata.get("parent_id") or child.extra_info.get("parent_id")
    vec_db.upsert_document(child_nodes)
    active_document = file.filename


    return {
        "filename": file.filename,
        "message": "File uploaded and processed successfully."
    }

@app.post("/ask")
async def ask_question(query: str):

    if active_document is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No document uploaded yet."
        )
    
    context = vec_db.hybrid_search(
        query,
        source=active_document
    )

    generated_response = llm.generate_response(query, context)
    
    return {"response": generated_response}
