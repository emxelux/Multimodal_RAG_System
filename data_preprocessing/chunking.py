"""
chunking.py — Parent-child semantic chunking.

Each PDF page becomes one "parent" (full text stored in a dict for later retrieval).
The page is further split by SemanticChunker into smaller "child" chunks that are
embedded and stored in Qdrant. Each child carries a parent_id reference.
"""

import uuid
from langchain_core.documents import Document
from langchain_experimental.text_splitter import SemanticChunker
from data_preprocessing.embedding import get_dense_embeddings


def chunk_pdf(pages: list, embeddings_model=None) -> tuple:
    """
    Split page dicts (from ingest.load_pdf) into child chunks.

    Args:
        pages:            output of ingest.load_pdf()
        embeddings_model: optional override; defaults to get_dense_embeddings()

    Returns:
        (child_chunks, parent_docs)
        child_chunks — list[Document]
        parent_docs  — dict {parent_id: page_text}
    """
    model = embeddings_model or get_dense_embeddings()
    splitter = SemanticChunker(model, breakpoint_threshold_type="percentile")

    child_chunks: list = []
    parent_docs: dict = {}

    for page in pages:
        page_text = page.get("text", "").strip()
        if not page_text:
            continue

        parent_id = str(uuid.uuid4())
        parent_docs[parent_id] = page_text

        for text in splitter.split_text(page_text):
            if not text.strip():
                continue
            child_chunks.append(
                Document(
                    page_content=text,
                    metadata={
                        "parent_id": parent_id,
                        "source": page["metadata"].get("file_path", ""),
                        "page": page["metadata"].get("page_number"),
                        "has_images": page["metadata"].get("has_images", False),
                        "has_tables": page["metadata"].get("has_tables", False),
                    },
                )
            )

    return child_chunks, parent_docs
