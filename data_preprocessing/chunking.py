from data_preprocessing.ingest import load_pdf, path
from langchain_text_splitters import RecursiveCharacterTextSplitter



def chunk_document(doc):
    text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=150,
    separators=["\n\n", "\n"]
)
    chunks = text_splitter.split_documents([doc])
    print("...................Chunked Document.........................")
    return chunks