from data_preprocessing.chunking import split_markdown_document
from data_preprocessing.ingest import build_documents, ingest_pdf
import json
import os
from pathlib import Path


file_path = "./document_files/MultimodalMachineLearning.pdf"
# sample_json_path = Path(__file__).resolve().parents[1] / "output1.json"

json_result = ingest_pdf(file_path)
documents = build_documents(
    json_result,
    source_name=os.path.basename(file_path)
)
# print(json_result)


nodes = split_markdown_document(documents)