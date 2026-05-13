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
    BASE_STORAGE = Path("/data")  # HuggingFace / Docker persistent storage
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


app = FastAPI()

@app.get("/")
def check_page_status():
    return {"response": "Hello! The app is working perfectly."}



def process_document(file_path: Path):
    global active_document

    file_path = file_path.resolve() 
    filename = file_path.name

    print(f"[INGESTION STARTED] {file_path}")


    docs = load_document(str(file_path))

    if not docs:
        raise RuntimeError("No text extracted from PDF (file may be scanned or corrupted)")

 
    parent_nodes = create_nodes(docs)

    if not parent_nodes:
        raise RuntimeError("Failed to create parent nodes from document")
    print("Parent Node created Sucessfully")

    for node in parent_nodes:
        parent_id = str(uuid.uuid4())
        node.metadata["parent_id"] = parent_id

        db.add_document(
            source=filename,
            parent_id=parent_id,
            parent_metadata=node.metadata,
            parent_content=node.get_content(),
        )

    child_nodes = chunk_nodes(parent_nodes)

    if not child_nodes:
        raise RuntimeError("No text chunks generated from document")
    print("Child node created successfully")

    # Ensure parent_id propagation
    for child in child_nodes:
        if "parent_id" not in child.metadata:
            child.metadata["parent_id"] = (
                child.metadata.get("parent_id")
                or getattr(child, "extra_info", {}).get("parent_id")
            )

    
    vec_db.upsert_document(child_nodes)
    print("Upsert Successful")


    active_document = filename

    print(f"[INGESTION COMPLETE] {filename}")



@app.post("/upload_pdf")
async def upload_file(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...)
):

    file_path = UPLOAD_DIR / file.filename

    try:
        # Save file
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)

        # Run ingestion in background
        background_tasks.add_task(process_document, file_path)

        return {
            "filename": file.filename,
            "message": "Upload successful. Processing started in background."
        }

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An Error Occured"
        )


@app.post("/ask")
async def ask_question(query: str):

    global active_document

    # this prevents early query before ingestion completes
    if active_document is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No document fully processed yet. Please wait for ingestion to complete."
        )

    try:
        # Retrieve relevant context
        context = vec_db.hybrid_search(
            query=query,
            source=active_document
        )

        if not context:
            raise HTTPException(
                status_code=404,
                detail="No relevant context found in document."
            )

        # Generate response
        generated_response = llm.generate_response(query, context)

        return {"response": generated_response}

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )