from langchain_qdrant import FastEmbedSparse
from langchain_huggingface import HuggingFaceEmbeddings



dense_embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5"
)

sparse_embeddings = FastEmbedSparse(
    model_name="Qdrant/bm25"
)
