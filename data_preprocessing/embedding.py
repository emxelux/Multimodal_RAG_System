from langchain_qdrant import FastEmbedSparse
from langchain_community.embeddings import HuggingFaceEmbeddings

dense_embedding = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5",
        cache_folder="/tmp/huggingface_cache")

sparse_embedding = FastEmbedSparse(model_name="Qdrant/bm25")
