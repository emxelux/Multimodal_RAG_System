import uuid
import shutil
import logging
from pathlib import Path
from typing import Optional
from contextlib import asynccontextmanager
from api.models import AskRequest, AskResponse
from fastapi import FastAPI, HTTPException, UploadFile, File, Query, status


logger = logging.getLogger(__name__)

_state: dict = {}


def get_vector_db():
    db = _state.get("vector_db")
    if db is None:
        raise HTTPException(status_code=503, detail="Vector DB not ready")
    return db


def get_llm():
    llm = _state.get("llm")
    if llm is None:
        raise HTTPException(status_code=503, detail="LLM not ready")
    return llm


def get_db():
    db = _state.get("db")
    if db is None:
        raise HTTPException(status_code=503, detail="Database not ready")
    return db


@asynccontextmanager
async def lifespan(app: FastAPI):
    from databases.database import Database
    from data_preprocessing.vector_db import VectorDB
    from llm.llm_connection import LLM

    logger.info("Starting up...")

    try:
        _state["db"] = Database()
        logger.info("PostgreSQL connected")
    except Exception as e:
        logger.error(f"Database init failed: {e}")

    try:
        _state["vector_db"] = VectorDB()
        logger.info("Qdrant connected")
    except Exception as e:
        logger.error(f"VectorDB init failed: {e}")

    try:
        _state["llm"] = LLM()
        logger.info("LLM ready")
    except Exception as e:
        logger.error(f"LLM init failed: {e}")

    yield
    _state.clear()


app = FastAPI(title="ChatPDF API", lifespan=lifespan)

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "document_files"
DATA_DIR.mkdir(exist_ok=True)





@app.get("/")
def homepage():
    return {"status": "ok", "message": "ChatPDF API is running"}


@app.post("/upload/", status_code=status.HTTP_201_CREATED)
async def upload_file(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only PDF files are allowed.",
        )

    from data_preprocessing.ingest import load_pdf
    from data_preprocessing.chunking import chunk_pdf

    file_path = DATA_DIR / file.filename
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

   pass

@app.get("/documents/")
def list_documents():
    return get_db().list_documents()


@app.get("/documents/{doc_id}")
def get_document(doc_id: int):
    doc = get_db().get_document(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    return doc


@app.delete("/documents/{doc_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_document(doc_id: int):
    if not get_db().delete_document(doc_id):
        raise HTTPException(status_code=404, detail="Document not found")


@app.post("/ask/", response_model=AskResponse)
def ask_question(
    question: Optional[str] = Query(default=None),
    body: Optional[AskRequest] = None,
):

    # Resolve question and optional fields from whichever source provided them
    resolved_question = (body.question if body else None) or question
    source = (body.source if body else None)
    conv_id = (body.conversation_id if body else None) or str(uuid.uuid4())

    if not resolved_question:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="'question' is required — pass as query param or JSON body field.",
        )

    history = get_db().get_conversation(conv_id)

    context = get_vector_db().search(
        query=resolved_question,
        top_k=5,
        source=source,
    )

    if not context:
        return AskResponse(
            answer="The answer to your question is not found in the uploaded document.",
            conversation_id=conv_id,
            sources=[],
        )

    answer = get_llm().generate_response(
        query=resolved_question,
        context=context,
        history=history,
    )

    get_db().add_message(conv_id, role="user", content=resolved_question)
    get_db().add_message(conv_id, role="assistant", content=answer)

    return AskResponse(
        answer=answer,
        conversation_id=conv_id,
        sources=[{"source": c["source"], "page": c["page"]} for c in context],
    )
