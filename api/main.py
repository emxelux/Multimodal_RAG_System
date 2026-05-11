import uuid
import os
from databases.database import Database, Document
from pathlib import Path
from data_preprocessing.ingest import load_document, create_nodes
from data_preprocessing.chunking import chunk_nodes
from data_preprocessing.embedding import dense_embedding, sparse_embedding
from data_preprocessing.vector_db import RAGVectorStore
from llm.llm_connection import LLM
from databases.database import Database
from fastapi import FastAPI, HTTPException, UploadFile, File

app = FastAPI()
UPLOAD_DIR = "document_files"
vec_db = RAGVectorStore()

@app.get("/")
def check_page_status():
    return {"response":"Hello! The app is working Perfectly"}

@app.post("/upload_pdf")
async def upload_file(file: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as f:
        content = await file.read()
        f.write(content)
    
    docs = load_document(file.filename)
    parent_nodes = create_nodes(docs)
    child_nodes = chunk_nodes(parent_nodes)
    vec_db.upsert_document(child_nodes)
    return {"response":"Successfully Uploaded PDF"}


@app.post("/ask")
async def ask_question(query: str):
    answer = vec_db.hybrid_search(query)
    return {"response": answer}