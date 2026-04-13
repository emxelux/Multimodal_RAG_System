import uuid
from typing import List, Optional
from dotenv import load_dotenv
from langchain_qdrant import QdrantVectorStore
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

from data_preprocessing.embedding import (
    dense_embeddings,
    sparse_embeddings,
)

load_dotenv()

qdrant_api_key = os.getenv("QDRANT_API_KEY")
class VectorDB:
    def __init__(self):
        self.collection_name = "chatpdf_knowledge"
        self.embeddings = dense_embeddings
        self.sparse_embeddings = sparse_embeddings

        self.client = QdrantClient(
           url= "https://bd1dcb05-82dd-48c8-a843-290ece2e38b3.us-west-2-0.aws.cloud.qdrant.io",
           api_key= qdrant_api_key
        )

        self._ensure_collection()
        
        self.vectorstore = QdrantVectorStore(
            client=self.client,
            collection_name=self.collection_name,
            embedding=self.embeddings,
            sparse_embedding=self.sparse_embeddings,
        )

    def _ensure_collection(self):
        collections = [
            c.name for c in self.client.get_collections().collections
        ]

        if self.collection_name not in collections:
            partial_embeddings = self.embeddings.embed_query("test")
            vector_size = len(partial_embeddings)

            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=vector_size,
                    distance=Distance.COSINE,
                ),
                sparse_vectors_config={
                    "sparse": SparseVectorParams()
                },
            )
            print(f"✅ Collection '{self.collection_name}' created with size {vector_size}")

    def build_index(
        self,
        documents: List[Document],
        source_name: str,
    ):
        self._ensure_collection()

        document_id = str(uuid.uuid4())
        
        for i, doc in enumerate(documents):
            doc.metadata.update({
                "source": source_name,
                "page": i + 1,
                "document_id": document_id,
            })
        self.vectorstore.add_documents(documents=documents)

        print(f"✅ Indexed {len(documents)} chunks from: {source_name}")

    def search(
        self,
        query: str,
        top_k: int = 5,
        source: Optional[str] = None,
    ):
        search_kwargs = {"k": top_k}

        if source:
            search_kwargs["filter"] = Filter(
                must=[
                    FieldCondition(
                        key="metadata.source",
                        match=MatchValue(value=source),
                    )
                ]
            )

        results = self.vectorstore.similarity_search(
            query=query,
            **search_kwargs,
        )

        return self._format_results(results)

    def _format_results(self, docs: List[Document]):
        output = []
        for d in docs:
            output.append({
                "content": d.page_content,
                "source": d.metadata.get("source"),
                "page": d.metadata.get("page"),
                "document_id": d.metadata.get("document_id"),
            })
        return output