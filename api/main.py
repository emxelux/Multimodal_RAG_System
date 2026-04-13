# from fastapi import FastAPI, HTTPException, UploadFile, File, status
# from databases.database import Database
# from data_preprocessing.vector_db import VectorDB
# from llm.llm_connection import LLM
# from data_preprocessing.ingest import load_pdf, clean_text
# from data_preprocessing.chunking import chunk_document
# from pathlib import Path
# import shutil

# app = FastAPI()


# ROOT_DIR = Path(__file__).resolve().parents[1]
# DOCUMENT_DIR = ROOT_DIR / "document_files"
# DOCUMENT_DIR.mkdir(exist_ok=True)
# my_client = VectorDB()
# llm = LLM()
# db = Database()



# @app.get("/")
# def homepage():
#     return {"response": "This is the homepage"}

# @app.post("/upload/")
# async def upload_file(file: UploadFile = File(...)):
#     if not file.filename.lower().endswith('.pdf'):
#         raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Only PDF files are allowed.")
#     file_path = DOCUMENT_DIR / file.filename
#     with open(file_path, "wb") as buffer:
#         shutil.copyfileobj(file.file, buffer)
#     text, source = load_pdf(str(file.filename))
#     text = clean_text(text, source)
#     chunks = chunk_document(text)
#     my_client.build_index(chunks, source_name = file.filename)
#     db.add_document(source=str(file.filename))
#     return {"file": file.filename}

# @app.post("/ask/")
# def ask_question(question: str):
#     result = my_client.search(question)
#     llm_response = llm.generate_response(query=question, result = result)
#     return {"answer": llm_response}

from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def root():
    return {"status": "working"}