import uuid
from databases.database import Database, Document
from data_preprocessing.ingest import load_document, create_nodes
from data_preprocessing.chunking import chunk_nodes
from data_preprocessing.embedding import dense_embedding, sparse_embedding
from data_preprocessing.vector_db import RAGVectorStore
from llm.llm_connection import LLM

llm = LLM()

db = Database()

store = RAGVectorStore(
    collection_name="chatpdf",
    dense_embedding=dense_embedding,
    sparse_embedding=sparse_embedding
)
source = "Emmanuel Abiodun Resume (Tech).pdf"

test_doc = load_document(source)
parent_nodes = create_nodes(test_doc)

for node in parent_nodes:
    parent_id = str(uuid.uuid4())
    node.metadata["parent_id"] = parent_id

    db.add_document(
        source=source,
        parent_id=parent_id,
        parent_metadata=node.metadata,
        parent_content=node.get_content(),
    )

child_nodes = chunk_nodes(parent_nodes)

for child in child_nodes:
    if "parent_id" not in child.metadata:
        child.metadata["parent_id"] = child.metadata.get("parent_id") or child.extra_info.get("parent_id")

store.upsert_document(child_nodes)
query = "What are the skill under Cloud & Deployment under Emmanuel Olanrewaju Tech stack?"
context = store.hybrid_search(query, top_k=3, source=source)


generated_answer = llm.generate_response(query, context)
print(generated_answer)


