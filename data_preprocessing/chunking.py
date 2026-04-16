"""
chunking.py — Parent-child semantic chunking.

Each PDF page becomes one "parent" (stored in a dict for full-context retrieval).
The page text is further split by SemanticChunker into smaller "child" chunks
that are embedded and stored in Qdrant.  Each child carries a parent_id so the
retriever can later fetch the richer parent text instead of just the snippet.
"""

import uuid
from langchain_core.documents import Document
from langchain_experimental.text_splitter import SemanticChunker


def chunk_pdf(pages: list, embeddings_model) -> tuple:
    """
    Split a list of page dicts (from ingest.load_pdf) into child chunks.

    Args:
        pages:            output of ingest.load_pdf()
        embeddings_model: a LangChain Embeddings object (used by SemanticChunker)

    Returns:
        (child_chunks, parent_docs)
        child_chunks  — list[Document]  small, embeddable chunks
        parent_docs   — dict {parent_id: page_text}  full page text keyed by UUID
    """
    splitter = SemanticChunker(
        embeddings_model,
        breakpoint_threshold_type="percentile",
    )

    child_chunks: list[Document] = []
    parent_docs: dict[str, str] = {}

    for page in pages:
        page_text = page.get("text", "").strip()
        if not page_text:
            continue

        parent_id = str(uuid.uuid4())
        parent_docs[parent_id] = page_text

        sub_texts = splitter.split_text(page_text)

        for text in sub_texts:
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
