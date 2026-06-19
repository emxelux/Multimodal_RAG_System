import os
from dotenv import load_dotenv
from llama_parse import LlamaParse
from langchain_core.documents import Document

load_dotenv()

os.environ["LLAMA_API_KEY"] = os.getenv("LLAMA_API_KEY")


from llama_parse import LlamaParse
import json

# Initialize the parser and set the result type to JSON
parser = LlamaParse(
    api_key=os.getenv("LLAMA_API_KEY"),
    result_type="json"  
)

def ingest_pdf(file_path:str):
    json_result = parser.get_json_result(file_path)
    return json_result




def build_documents(parsed_json, source_name):
    documents = []

    pages = parsed_json[0]["pages"]

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