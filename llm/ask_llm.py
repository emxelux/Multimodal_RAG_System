"""
ask_llm.py — Quick CLI test: upload a PDF and ask a question.

Usage (from project root):
    python -m llm.ask_llm
"""

from data_preprocessing.ingest import load_pdf
from data_preprocessing.chunking import chunk_pdf
from data_preprocessing.embedding import dense_embeddings
from data_preprocessing.vector_db import VectorDB
from llm.llm_connection import LLM

PDF_FILE = "about.pdf"   # place your PDF in document_files/
QUESTION = "What is this document about?"

def main():
    print(f"Loading '{PDF_FILE}' ...")
    pages = load_pdf(PDF_FILE)
    print(f"  → {len(pages)} pages")

    child_chunks, parent_docs = chunk_pdf(pages, dense_embeddings)
    print(f"  → {len(child_chunks)} chunks, {len(parent_docs)} parents")

    db = VectorDB()
    db.build_index(child_chunks, source_name=PDF_FILE)

    results = db.search(QUESTION, top_k=5)
    print(f"\nTop {len(results)} results retrieved.")

    llm = LLM()
    answer = llm.generate_response(query=QUESTION, context=results)
    print(f"\nQuestion: {QUESTION}\nAnswer  : {answer}")

if __name__ == "__main__":
    main()
