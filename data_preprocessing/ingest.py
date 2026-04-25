import os
import dotenv
from pathlib import Path
from llama_parse import LlamaParse
from llama_index.core import Document
from llama_index.core.node_parser import MarkdownNodeParser

dotenv.load_dotenv()



ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "document_files"
ASSETS_DIR = ROOT_DIR / "assets"
ASSETS_DIR.mkdir(exist_ok=True)


parser = LlamaParse(
    result_type="markdown",
    api_key=os.getenv("LLAMA_API_KEY"),
)


def load_document(file_path:str):
    documents = parser.load_data(f"{DATA_DIR}/{file_path}")
    docs = []
    for doc in documents:
        page_num = doc.metadata.get("page_label") or doc.metadata.get("page_number", "1")
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
    nodes = parser.get_nodes_from_documents(documents)
    return nodes


test_doc = load_document("skills_guide.pdf")

print(create_nodes(test_doc))