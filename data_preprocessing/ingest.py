import os
import dotenv
from pathlib import Path
from llama_parse import LlamaParse
from llama_index.core import Document
from llama_index.core.node_parser import MarkdownNodeParser

dotenv.load_dotenv()



if os.path.exists("/data"):
    BASE_STORAGE = Path("/data")  # HuggingFace Persistent Storage
else:
    BASE_STORAGE = Path(__file__).resolve().parent

UPLOAD_DIR = BASE_STORAGE / "document_files"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

parser = LlamaParse(
    result_type="markdown",
    api_key=os.getenv("LLAMA_API_KEY"),
)


def load_document(file_path:str):
    documents = parser.load_data(f"{UPLOAD_DIR}/{file_path}")
    docs = []
    for doc in documents:
        page_num = doc.metadata.get("page")
        text = doc.get_content()
        docs.append({
            "page_num": page_num,
            "source": file_path,
            "content": text,
        })
    return docs

def create_nodes(docs):
    documents = [
        Document(text=doc["content"], metadata={
            "page": doc["page_num"],
            "source": doc["source"]
        })
        for doc in docs
    ]
    parser = MarkdownNodeParser()
    parent_nodes = parser.get_nodes_from_documents(documents)
    return parent_nodes


