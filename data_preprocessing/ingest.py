import re
import puremagic
import pymupdf4llm
from langchain_core.documents import Document
from pathlib import Path
import fitz
import pdfplumber
from pypdf import PdfReader

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "document_files"
path = DATA_DIR



def load_pdf(file_path: str):
    file_path = f"{path}/{file_path}"
    pages = pymupdf4llm.to_markdown(
        file_path, 
        page_chunks=True, 
        write_images=True, 
        image_path="assets"
    )
    return pages
    


