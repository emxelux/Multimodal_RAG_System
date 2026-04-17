"""
embedding.py — Lazy-loaded embedding singletons.

Models are only downloaded/instantiated the first time get_*() is called,
not at import time. This prevents blocking the uvicorn startup process
while model files are fetched from HuggingFace.
"""

from functools import lru_cache
from langchain_qdrant import FastEmbedSparse
from langchain_community.embeddings import HuggingFaceEmbeddings


@lru_cache(maxsize=1)
def get_dense_embeddings() -> HuggingFaceEmbeddings:
    """Return the shared dense embedding model (downloaded once, cached forever)."""
    return HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5",
        cache_folder="/tmp/huggingface_cache")


@lru_cache(maxsize=1)
def get_sparse_embeddings() -> FastEmbedSparse:
    """Return the shared sparse BM25 embedding model (downloaded once, cached forever)."""
    return FastEmbedSparse(model_name="Qdrant/bm25")
