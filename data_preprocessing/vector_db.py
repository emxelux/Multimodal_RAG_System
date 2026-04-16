"""
vector_db.py — Qdrant hybrid (dense + sparse) vector store wrapper.

Uses:
  - Dense  : BAAI/bge-small-en-v1.5  (HuggingFaceEmbeddings)
  - Sparse : Qdrant/bm25              (FastEmbedSparse)
  - Mode   : HYBRID                   (RRF fusion)

langchain_qdrant requires the sparse vector field to be named "langchain-sparse"
exactly. The collection is validated on startup and recreated if the schema
is wrong (e.g. created by an older version of this code).
"""

import uuid
import os
from typing import List, Optional
from dotenv import load_dotenv

from langchain_qdrant import QdrantVectorStore, RetrievalMode
from langchain_core.documents import Document
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    SparseVectorParams,
    Filter,
    FieldCondition,
    MatchValue,
)

from data_preprocessing.embedding import get_dense_embeddings, get_sparse_embeddings

load_dotenv()

SPARSE_VECTOR_NAME = "langchain-sparse"


class VectorDB:
    def __init__(self):
        self.collection_name = "chatpdf_knowledge"

        self.client = QdrantClient(
            url="https://bd1dcb05-82dd-48c8-a843-290ece2e38b3.us-west-2-0.aws.cloud.qdrant.io",
            api_key=os.getenv("QDRANT_API_KEY"),
        )

        # Models are loaded lazily here (first call triggers download)
        self._ensure_collection()

        self.vectorstore = QdrantVectorStore(
            client=self.client,
            collection_name=self.collection_name,
            embedding=get_dense_embeddings(),
            sparse_embedding=get_sparse_embeddings(),
            sparse_vector_name=SPARSE_VECTOR_NAME,
            retrieval_mode=RetrievalMode.HYBRID,
        )

    # ------------------------------------------------------------------
    # collection management
    # ------------------------------------------------------------------

    def _ensure_collection(self):
        existing = [c.name for c in self.client.get_collections().collections]

        if self.collection_name in existing:
            info = self.client.get_collection(self.collection_name)
            sparse_names = list((info.config.params.sparse_vectors or {}).keys())
            if SPARSE_VECTOR_NAME not in sparse_names:
                print(
                    f"⚠️  Collection '{self.collection_name}' missing sparse field "
                    f"'{SPARSE_VECTOR_NAME}' — recreating."
                )
                self.client.delete_collection(self.collection_name)
            else:
                return

        vector_size = len(get_dense_embeddings().embed_query("test"))
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
            sparse_vectors_config={SPARSE_VECTOR_NAME: SparseVectorParams()},
        )
        print(f"✅ Collection '{self.collection_name}' created (dim={vector_size})")

    # ------------------------------------------------------------------
    # indexing
    # ------------------------------------------------------------------

    def build_index(self, documents: List[Document], source_name: str) -> str:
        if not documents:
            print(f"⚠️  No chunks to index for: {source_name}")
            return ""

        self._ensure_collection()
        document_id = str(uuid.uuid4())

        for doc in documents:
            doc.metadata.setdefault("source", source_name)
            doc.metadata["document_id"] = document_id

        self.vectorstore.add_documents(documents=documents)
        print(f"✅ Indexed {len(documents)} chunks from: {source_name}")
        return document_id

    # ------------------------------------------------------------------
    # retrieval
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        top_k: int = 5,
        source: Optional[str] = None,
    ) -> List[dict]:
        search_kwargs: dict = {"k": top_k}

        if source:
            search_kwargs["filter"] = Filter(
                must=[
                    FieldCondition(
                        key="metadata.source",
                        match=MatchValue(value=source),
                    )
                ]
            )

        results = self.vectorstore.similarity_search(query=query, **search_kwargs)
        return self._format_results(results)

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _format_results(self, docs: List[Document]) -> List[dict]:
        return [
            {
                "content": d.page_content,
                "source": d.metadata.get("source"),
                "page": d.metadata.get("page"),
                "document_id": d.metadata.get("document_id"),
                "has_images": d.metadata.get("has_images", False),
                "has_tables": d.metadata.get("has_tables", False),
            }
            for d in docs
        ]
