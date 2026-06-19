from data_preprocessing.ingest import ingest_pdf, build_documents
from data_preprocessing.chunking import split_markdown_document
import os


file_path = "./document_files/MultimodalMachineLearning.pdf"

json_result = ingest_pdf(file_path)

documents = build_documents(
    json_result,
    source_name=os.path.basename(file_path)
)

print(split_markdown_document(documents=documents))