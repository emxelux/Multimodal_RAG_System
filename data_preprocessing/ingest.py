import os
from langchain_core.documents import Document
from loguru import logger
from dotenv import load_dotenv

load_dotenv()



try:
    from llama_parse import LlamaParse
except ModuleNotFoundError:
    LlamaParse = None


llama_api_key = os.getenv("LLAMA_API_KEY")
if llama_api_key:
    os.environ["LLAMA_API_KEY"] = llama_api_key

def ingest_pdf(file_path:str):
    
    parser = LlamaParse(
        api_key=llama_api_key,
        result_type="json"
    )
    json_result = parser.get_json_result(file_path)
    if json_result:
        logger.info("JSON CREATED SUCCESFFULLY FROM DOCUMENTS")
    return json_result


# print(ingest_pdf("document_files/MultimodalMachineLearning.pdf"))


def build_documents(parsed_json, source_name):
    documents = []
    if not parsed_json: #or not isinstance(parsed_json, list):
        logger.info(f"Warning: parsed_json is empty or invalid for {source_name}")
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
