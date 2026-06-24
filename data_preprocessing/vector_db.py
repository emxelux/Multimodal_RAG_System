import os
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
# Fixed: Imported FastEmbedSparse and RetrievalMode cleanly from langchain_qdrant
from langchain_qdrant import QdrantVectorStore, RetrievalMode, FastEmbedSparse
from langchain_cohere import CohereRerank
from qdrant_client import QdrantClient, models
from functools import lru_cache
# Load environment variables
load_dotenv()

# Map Gemini credentials cleanly
if "GEMINI_API_KEY" in os.environ:
    os.environ["GOOGLE_API_KEY"] = os.environ["GEMINI_API_KEY"]


@lru_cache(maxsize=1)
def get_vector_store():

# 1. Initialize Dense and Sparse Embedding Models
    dense_embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")
    sparse_embeddings = FastEmbedSparse(model_name="Qdrant/bm25")

    # 2. Setup Qdrant Client 
    client = QdrantClient(path="./qdrant_storage") 
    collection_name = "chatypdf_collection"

    # Fixed: Explicitly provision the collection with structural layout for both dimensions
    client.create_collection(
        collection_name=collection_name,
        # force_recreate = True,
        vectors_config={
            "dense": models.VectorParams(
                size=3072,  # size of models/embedding-001
                distance=models.Distance.COSINE
            )
        },
        sparse_vectors_config={
            "sparse": models.SparseVectorParams(
                index=models.SparseIndexParams(on_disk=False)
            )
        }
    )

    # 3. Configure Vector Store with Proper Hybrid Integration
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
        chunk.metadata["user_id"] = user_id
        chunk.metadata["source"] = source_document
        docs_to_upsert.append(chunk)
        
    vector_store.add_documents(docs_to_upsert)
    print(f"Successfully indexed {len(docs_to_upsert)} chunks.")


# --- STEP 5: RETRIEVAL WITH METADATA FILTERING ---
def retrieve_context(query, user_id, source_document, top_k=10):
    vector_store = get_vector_store()
    qdrant_filter = models.Filter(
        must=[
            models.FieldCondition(
                key="metadata.user_id",
                match=models.MatchValue(value=str(user_id)),
            ),
            models.FieldCondition(
                key="metadata.source",
                match=models.MatchValue(value=source_document),
            ),
        ]
    )

    results = vector_store.similarity_search(
        query=query,
        k=top_k,
        filter=qdrant_filter,
    )
    return results


# --- STEP 6: COHERE RERANKING ---
def rerank_results(query, documents, top_n=3):
    if not documents:
        return []
        
    reranker = CohereRerank(model="rerank-v3.5", top_n=top_n)
    reranked_docs = reranker.compress_documents(documents=documents, query=query)
    return reranked_docs


# --- EXECUTION ---

# from data_preprocessing.test_ingest import nodes
# # from data_preprocessing.chunking import split_markdown_document
# print(nodes[:3])
# test_user = "user1"
# test_file = "DATA-ANALYSIS-REPORT-TEAM-5.pdf"

# # 1. Ingest Data
# upsert_split_documents(nodes, test_user, test_file)
# print(f"Upserted {len(nodes)} to vector database")
# # 2. Hybrid Retrieve with Filter
# query_str = "What are the differences between early fusion and late fusion"
# raw_results = retrieve_context(query_str, test_user, test_file)
# print(f"\nRetrieved {len(raw_results)} documents from Hybrid Search.")

# # 3. Apply Re-ranking via Cohere
# final_context = rerank_results(query_str, raw_results)
# print(f"\n--- Top {len(final_context)} Reranked Results ---")
# doc_context = []
# for idx, doc in enumerate(final_context):
#     doc_context.append(f"[{idx + 1}] (Score: {doc.metadata.get('relevance_score', 'N/A')}) \n\n {doc.page_content}\n\n Page: {doc.metadata.get("page")}...")