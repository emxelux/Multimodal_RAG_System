from typing import List, Optional
from qdrant_client import QdrantClient
from data_preprocessing.embedding import dense_embedding, sparse_embedding
from qdrant_client.models import Filter, FieldCondition, MatchValue
from llama_index.core.vector_stores import MetadataFilters, ExactMatchFilter, FilterOperator
from databases.database import Database
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.core.schema import BaseNode, NodeWithScore
from llama_index.postprocessor.sbert_rerank import SentenceTransformerRerank
from dotenv import load_dotenv
import os


db = Database()
load_dotenv()

class RAGVectorStore:
    def __init__(
        self,
        collection_name: str = "pdf_collection",
        dense_embedding = dense_embedding,
        sparse_embedding = sparse_embedding,
        storage_path: str = "https://bd1dcb05-82dd-48c8-a843-290ece2e38b3.us-west-2-0.aws.cloud.qdrant.io",
        reranker_model: str = "BAAI/bge-reranker-base",
        reranker_top_n: int = 5,
    ) -> None:
        self.collection_name = collection_name
        self.dense_embedding = dense_embedding
        self.sparse_embedding = sparse_embedding
        self.client = QdrantClient(url=storage_path, api_key = os.environ.get("QDRANT_API_KEY"))
        
        self.client.create_payload_index(
        collection_name=self.collection_name,
        field_name="source",
        field_schema="keyword"
    )
        
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

    def _get_or_create_index(self) -> VectorStoreIndex:
        if self.index is None:
            self.index = VectorStoreIndex.from_vector_store(
                vector_store=self.vector_store,
                embed_model=self.dense_embedding,
            )
        return self.index




    def hybrid_search(self, query: str, source: Optional[str] = None, top_k: int = 5) -> List[NodeWithScore]:
        try:
            index = self._get_or_create_index()
            
            filters = None
            if source:
                filters = MetadataFilters(
                    filters=[
                        ExactMatchFilter(
                            key="source",
                            value=source,
                        )
                    ]
                )
            
            retriever = index.as_retriever(
                similarity_top_k=top_k * 3,
                vector_store_query_mode="hybrid",
                filters=filters,
            )
            nodes = retriever.retrieve(query)
            reranked_nodes = self.reranker.postprocess_nodes(
                nodes,
                query_str=query,
            )
            return [db.get_parent_content(n.metadata.get("parent_id")) for n in reranked_nodes]
        except Exception as e:
            raise RuntimeError(f"Hybrid search failed: {str(e)}")