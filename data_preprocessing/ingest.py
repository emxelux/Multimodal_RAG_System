import re
import puremagic
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from pathlib import Path
import fitz
import pdfplumber
from pypdf import PdfReader

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "document_files"
path = DATA_DIR
def get_full_path(file_path: str) -> Path:
    full_path = DATA_DIR / file_path
    if not full_path.exists():
        raise FileNotFoundError(f"File not found: {full_path}")
    return full_path

def detect_filetype(file_path):
    try:
        return puremagic.from_file(str(get_full_path(file_path)))
    except Exception as e:
        return e

def is_text_valid(text: str) -> bool:
    if not text:
        return False
    return len(text.strip()) >= 500

def load_with_pymupdf(file_path: Path) -> str:
    text = ""
    doc = fitz.open(str(file_path))
    for page in doc:
        text += page.get_text("text")
    doc.close()
    return text

def load_with_pdfplumber(file_path: Path) -> str:
    text = ""
    with pdfplumber.open(str(file_path)) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text

def load_with_pypdf(file_path: Path) -> str:
    text = ""
    reader = PdfReader(str(file_path))
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text

def load_pdf(file_path: str):
    full_path = get_full_path(file_path)
    loaders = [
        ("PyMuPDF", load_with_pymupdf),
        ("pdfplumber", load_with_pdfplumber),
        ("pypdf", load_with_pypdf),
    ]

    for name, loader in loaders:
        try:
            print(f"Trying loader: {name}")
            text = loader(full_path)

            if is_text_valid(text):
                print(f"✅ Success with {name}")
                return text, file_path

            print(f"⚠️ {name} extracted low-quality text")

        except Exception as e:
            print(f"OOps! {name} failed:", e)

    raise ValueError("All PDF loaders failed.")

def clean_text(text: str, file_path: str) -> Document:
    text = re.sub(r'\n\n+', '\n\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    return Document(page_content=text, metadata={"source": file_path})
