import os
from langchain_core.documents import Document
import logging
from dotenv import load_dotenv


import os
import psutil

process = psutil.Process(os.getpid())

def log_memory(stage: str):
    memory_mb = process.memory_info().rss / (1024 * 1024)
    logger.info(f"[MEMORY] {stage}: {memory_mb:.2f} MB")


load_dotenv()

logger = logging.getLogger(__name__)

from llama_cloud import LlamaCloud


llama_api_key = os.getenv("LLAMA_API_KEY")
if llama_api_key:
    os.environ["LLAMA_API_KEY"] = llama_api_key

def ingest_pdf(file_path:str):
    logger.info("===================== STARTING INGESTING PDF =======================================")

    log_memory(">>>>>>>>>>>>>>>>>>>>>>>>> Before llama cloud api <<<<<<<<<<<<<<<<<<<<<<<")
    client = LlamaCloud(api_key=os.getenv("LLAMA_API_KEY"))
    log_memory(">>>>>>>>>>>>>>>>>>>>>>>>> Before creating file <<<<<<<<<<<<<<<<<<<<<<<")
    file = client.files.create(
        file = file_path,
        purpose = "parse"
    )
    log_memory(">>>>>>>>>>>>>>>>>>>>>>>>> After Creating file <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
    logger.info("======================  FILE CREATED SUCCESSFULLY ===========================")
    log_memory(">>>>>>>>>>>>>>>>>>>>>>>>>>>> Before Parsing <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
    result = client.parsing.parse(
    file_id=file.id,
    tier="agentic",
    version="latest",
    output_options={
        "markdown": {"tables": {"output_tables_as_markdown": True}},
    },
    processing_options={
        "ocr_parameters": {"languages": ["en"]},
    },
)
    log_memory(">>>>>>>>>>>>>>>>>>>>>>> After Parsing <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
    logger.info("========================== MARKDOWN RESULT CREATED SUCCESSFULLY ===================================")
    return result.markdown.pages



def build_documents(ingested_document, source_name):
    logger.info("================= STARTING BUILDING DOCUMENT =============================")
    documents = []
    if not ingested_document:
        logger.warning("====================== No ingested document data provided =========================================")
        return documents

    for pages in ingested_document:
        page_num = pages.page_number if hasattr(pages, 'page_number') else None
        markdown_text = pages.markdown if hasattr(pages, 'markdown') else str(pages)
        documents.append(
            Document(
                page_content=markdown_text,
                metadata={
                    "source": source_name,
                    "page": page_num
                }
            )
        )
        logger.info("======================== DOCUMENT BUILDING DONE SUCCESSFULLY ============================")

    return documents

