from typing import List, Optional
from qdrant_client import QdrantClient
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.core.schema import BaseNode, NodeWithScore
from llama_index.postprocessor.sbert_rerank import SentenceTransformerRerank
from dotenv import load_dotenv
import os

load_dotenv()

class RAGVectorStore:
    def __init__(
        self,
        collection_name: str,
        dense_embedding,
        sparse_embedding: Optional[any] = None,
        storage_path: str = "./qdrant_storage",
        reranker_model: str = "BAAI/bge-reranker-base",
        reranker_top_n: int = 5,
    ) -> None:
        self.collection_name = collection_name
        self.dense_embedding = dense_embedding
        self.sparse_embedding = sparse_embedding
        self.client = QdrantClient(path=storage_path)
        
        self.vector_store = QdrantVectorStore(
            client=self.client,
            collection_name=self.collection_name,
            enable_hybrid=True,
        )
        
        self.storage_context = StorageContext.from_defaults(
            vector_store=self.vector_store
        )
        self.index = None
        
        self.reranker = SentenceTransformerRerank(
            model=reranker_model,
            top_n=reranker_top_n,
        )

    def upsert_document(self, nodes: List[BaseNode]) -> VectorStoreIndex:
        if not nodes:
            raise ValueError("nodes cannot be empty")
        
        try:
            self.index = VectorStoreIndex(
                nodes=nodes,
                storage_context=self.storage_context,
                embed_model=self.dense_embedding,
            )
            return self.index
        except Exception as e:
            raise RuntimeError(f"Failed to upsert documents: {str(e)}")

    def hybrid_search(self, query: str, top_k: int = 5) -> List[NodeWithScore]:
        if self.index is None:
            raise ValueError(
                "Index not initialized. Run upsert_document() first."
            )
        
        try:
            retriever = self.index.as_retriever(
                similarity_top_k=top_k * 3,
                vector_store_query_mode="hybrid",
            )
            nodes = retriever.retrieve(query)
            reranked_nodes = self.reranker.postprocess_nodes(
                nodes,
                query_str=query,
            )
            return reranked_nodes
        except Exception as e:
            raise RuntimeError(f"Hybrid search failed: {str(e)}")