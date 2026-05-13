import os
import uuid
from pathlib import Path
from fastapi import FastAPI, HTTPException, UploadFile, File, status, BackgroundTasks
from databases.database import Database
from data_preprocessing.ingest import load_document, create_nodes
from data_preprocessing.chunking import chunk_nodes
from data_preprocessing.embedding import dense_embedding, sparse_embedding
from data_preprocessing.vector_db import RAGVectorStore
from llm.llm_connection import LLM




if os.path.exists("/data"):
    BASE_STORAGE = Path("/data")
else:
    BASE_STORAGE = Path(__file__).resolve().parent

UPLOAD_DIR = BASE_STORAGE / "document_files"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)



app = FastAPI()

vec_db = RAGVectorStore(
    dense_embedding=dense_embedding,
    sparse_embedding=sparse_embedding
)

db = Database()
llm = LLM()


active_document = None


processing_status = {}


@app.get("/")
def check_page_status():
    return {"response": "Hello! The app is working perfectly."}



def process_document(file_path: Path):
 
    global active_document, processing_status

    file_path = file_path.resolve() 
    filename = file_path.name
    
    # Track status
    processing_status[filename] = "processing"
    print(f"[INGESTION STARTED] {file_path}")

    try:
        docs = load_document(str(file_path))

        if not docs:
            raise RuntimeError("No text extracted from PDF (file may be scanned or corrupted)")

     
        parent_nodes = create_nodes(docs)

        if not parent_nodes:
            raise RuntimeError("Failed to create parent nodes from document")
        print("Parent nodes created successfully")

        for node in parent_nodes:
            parent_id = str(uuid.uuid4())
            node.metadata["parent_id"] = parent_id
            if not node.extra_info:
                node.extra_info = {}
            node.extra_info["parent_id"] = parent_id

            db.add_document(
                source=filename,
                parent_id=parent_id,
                parent_metadata=node.metadata,
                parent_content=node.get_content(),
            )

        child_nodes = chunk_nodes(parent_nodes)

        if not child_nodes:
            raise RuntimeError("No text chunks generated from document")
        print("Child nodes created successfully")

        for i, child in enumerate(child_nodes):
            if "parent_id" not in child.metadata or not child.metadata["parent_id"]:
                if parent_nodes:
                    parent_id = parent_nodes[0].metadata.get("parent_id")
                    if parent_id:
                        child.metadata["parent_id"] = parent_id
                        if not child.extra_info:
                            child.extra_info = {}
                        child.extra_info["parent_id"] = parent_id
                        print(f"[RECOVERY] Restored parent_id for child node {i}")

        
        vec_db.upsert_document(child_nodes)
        print("Upsert successful")

        active_document = filename
        processing_status[filename] = "completed"

        print(f"[INGESTION COMPLETE] {filename}")
    
    except Exception as e:
        error_msg = str(e)
        processing_status[filename] = f"failed: {error_msg}"
        print(f"[INGESTION FAILED] {filename}: {error_msg}")
        raise



@app.post("/upload_pdf")
async def upload_file(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...)
):
    file_path = UPLOAD_DIR / file.filename

    try:
 
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)

        background_tasks.add_task(process_document, file_path)

        return {
            "filename": file.filename,
            "message": "Upload successful. Processing started in background.",
            "status_endpoint": f"/status/{file.filename}"
        }

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to upload file: {str(e)}"
        )

@app.get("/status/{filename}")
async def get_processing_status(filename: str):
    
    status_value = processing_status.get(filename, "not_found")
    return {
        "filename": filename,
        "status": status_value
    }


@app.post("/ask")
async def ask_question(query: str):
    global active_document

    if active_document is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No document processed yet. Please upload a PDF first and check /status/{filename} to confirm processing is complete."
        )

    try:

        context = vec_db.hybrid_search(
            query=query,
            source=active_document
        )

        if not context or all(not chunk for chunk in context):
            raise HTTPException(
                status_code=404,
                detail=f"No relevant context found for your query in '{active_document}'. Try rephrasing your question or ensure the document contains relevant information."
            )

        # Generate response
        generated_response = llm.generate_response(query, context)

        return {"response": generated_response}

    except HTTPException:
        raise
    except Exception as e:
        print(f"[ERROR] Query processing failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing query: {str(e)}"
        )
