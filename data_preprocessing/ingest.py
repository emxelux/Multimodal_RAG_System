import os
import dotenv
from pathlib import Path
from llama_parse import LlamaParse
from llama_index.core import Document
from llama_index.core.node_parser import MarkdownNodeParser

dotenv.load_dotenv()

# Storage root
if os.path.exists("/data"):
    BASE_STORAGE = Path("/data")
else:
    BASE_STORAGE = Path(__file__).resolve().parent

UPLOAD_DIR = BASE_STORAGE / "document_files"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

parser = LlamaParse(
    result_type="markdown",
    api_key=os.getenv("LLAMA_API_KEY"),
)



def load_document(file_path: str):
    file_path = str(file_path)

    # 🔥 FIX: DO NOT prepend UPLOAD_DIR again
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    documents = parser.load_data(file_path)

    if not documents:
        raise ValueError("LlamaParse returned empty document (likely scanned PDF or bad path)")

    docs = []
    for doc in documents:
        docs.append({
            "page_num": doc.metadata.get("page"),
            "source": file_path,
            "content": doc.get_content(),
        })

    return docs


def create_nodes(docs):
    documents = [
        Document(
            text=doc["content"],
            metadata={
                "page": doc["page_num"],
                "source": doc["source"]
            }
        )
        for doc in docs
    ]

    parser = MarkdownNodeParser()
    return parser.get_nodes_from_documents(documents)