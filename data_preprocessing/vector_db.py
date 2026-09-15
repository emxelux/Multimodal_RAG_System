import sys
import os

from functools import lru_cache
from dotenv import load_dotenv

from langchain_qdrant import QdrantVectorStore, RetrievalMode, FastEmbedSparse
from langchain_cohere import CohereRerank
import logging
from qdrant_client import QdrantClient, models
from qdrant_client.models import PayloadSchemaType

# Configure module-level logger cleanly

load_dotenv()

logger = logging.getLogger(__name__)



@lru_cache(maxsize=1)
def get_vector_store():
    from langchain_huggingface import HuggingFaceEndpointEmbeddings

    dense_embeddings = HuggingFaceEndpointEmbeddings(
        model="BAAI/bge-large-en-v1.5",
        huggingfacehub_api_token=os.environ["HF_TOKEN"],
    )
    sparse_embeddings = FastEmbedSparse(model_name="Qdrant/bm25")
    
    dense_dim = 1024

    # 2. Setup Qdrant Client 
    client = QdrantClient(
        url=os.getenv("QDRANT_ENDPOINT"),
        api_key=os.getenv("QDRANT_API_KEY"),
        timeout=120,
        # path="./qdrant_storage"
    ) 
    collection_name = "chatmypdf_"

    if not client.collection_exists(collection_name=collection_name):
        logger.info("CREATING COLLECTION: %s", collection_name)
        
        client.create_collection(
            collection_name=collection_name,
            vectors_config={
                "dense": models.VectorParams(
                    size=dense_dim,
                    distance=models.Distance.COSINE
                )
            },
            sparse_vectors_config={
                "sparse": models.SparseVectorParams(
                    index=models.SparseIndexParams(on_disk=False)
                )
            }
        )
        logger .info("COLLECTION CREATED SUCCESSFULLY")
        
        client.create_payload_index(
            collection_name=collection_name,
            field_name="metadata.user_id",
            field_schema=PayloadSchemaType.KEYWORD,
        )
        client.create_payload_index(
            collection_name=collection_name,
            field_name="metadata.source",
            field_schema=PayloadSchemaType.KEYWORD,
        )

   
    return QdrantVectorStore(
        client=client,
        collection_name=collection_name,
        embedding=dense_embeddings,
        sparse_embedding=sparse_embeddings,
        retrieval_mode=RetrievalMode.HYBRID,
        vector_name="dense",
        sparse_vector_name="sparse"
    )


# --- STEP 4: UPSERT DOCUMENTS WITH METADATA ---
def upsert_split_documents(markdown_nodes, user_id, source_document):
    vector_store = get_vector_store()
    docs_to_upsert = []
    
    for chunk in markdown_nodes:
        chunk.metadata["user_id"] = str(user_id)
        chunk.metadata["source"] = source_document
        docs_to_upsert.append(chunk)

    logger.info("=====================================  ABOUT TO UPSERT DOCUMENT TO VECTOR DATABASE  =============================")
    try:
        vector_store.add_documents(docs_to_upsert)
        logger.info(
    " ============================ INDEXING → user_id=%r | source=%r  =================================",
    str(user_id),
    str(source_document)
)
        logger.info("SUCCESSFULLY INDEXED %d CHUNKS.", len(docs_to_upsert))
    except Exception as e:
        logger.error("ERROR UPSERTING TO VECTORDB: %s", e)
        raise


# --- STEP 5: RETRIEVAL WITH METADATA FILTERING ---
def retrieve_context(query, user_id, source_document, top_k=10):
    vector_store = get_vector_store()
    
    # Matching payload keys natively flat-stored by langchain-qdrant
    qdrant_filter = models.Filter(
        must=[
            models.FieldCondition(
                key="metadata.user_id",
                match=models.MatchValue(value=str(user_id)),
            ),
            models.FieldCondition(
                key="metadata.source",
                match=models.MatchValue(value=str(source_document)),
            ),
        ]
    )

    results = vector_store.similarity_search(
        query=query,
        k=top_k,
        filter=qdrant_filter,
    )
    logger.info(
    "RETRIEVING → user_id=%r | source=%r",
    str(user_id),
    str(source_document)
)
    if not results:
        logger.error("The retrieval did not retrieve any result")
    logger.info(f"======= RETRIEVAL RESULT \n {results} \n ========== ")
    for i, doc in enumerate(results[:5]):
        logger.info(
        "RESULT %d METADATA: %r",
        i,
        doc.metadata
    )
    return results


# --- STEP 6: COHERE RERANKING ---
def rerank_results(query, documents, top_n=3):
    if not documents:
        return []
        
    reranker = CohereRerank(model="rerank-v3.5", top_n=top_n)
    reranked_docs = reranker.compress_documents(documents=documents, query=query)
    logger.info(f"========== RERANKED CONTEXT DOCUMENT\n {reranked_docs} \n ===================")
    return reranked_docs