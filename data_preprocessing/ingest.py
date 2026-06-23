import os
from langchain_core.documents import Document

try:
    from dotenv import load_dotenv
except ModuleNotFoundError:
    load_dotenv = None

try:
    from llama_parse import LlamaParse
except ModuleNotFoundError:
    LlamaParse = None

if load_dotenv is not None:
    load_dotenv()

llama_api_key = os.getenv("LLAMA_API_KEY")
if llama_api_key:
    os.environ["LLAMA_API_KEY"] = llama_api_key

def ingest_pdf(file_path:str):
    if LlamaParse is None:
        raise ImportError(
            "llama_parse is not installed. Install project dependencies or use the local sample JSON fixture."
        )

    parser = LlamaParse(
        api_key=llama_api_key,
        result_type="json"
    )
    json_result = parser.get_json_result(file_path)
    return json_result




def build_documents(parsed_json, source_name):
    documents = []

    # 1. Guard clause: Check if parsed_json is empty, None, or not a list
    if not parsed_json or not isinstance(parsed_json, list):
        print(f"Warning: parsed_json is empty or invalid for {source_name}")
        return documents

    # 2. Safe extraction using .get() just in case "pages" is missing
    first_item = parsed_json[0]
    pages = first_item.get("pages", [])

    for page in pages:
        page_num = page.get("page", 0)
        markdown_text = page.get("md", "")

        documents.append(
            Document(
                page_content=markdown_text,
                metadata={
                    "source": source_name,
                    "page": page_num
                }
            )
        )

    return documents


# file_path = "./document_files/MultimodalMachineLearning.pdf"

# json_result = ingest_pdf(file_path)

# documents = build_documents(
#     json_result,
#     source_name=os.path.basename(file_path)
# )

# print(documents[2])