import uuid
from langchain_core.documents import Document
from langchain_experimental.text_splitter import SemanticChunker


def chunk_pdf(pages, embeddings_model):
    semantic_splitter = SemanticChunker(embeddings_model, breakpoint_threshold_type="percentile")
    
    child_chunks = []
    parent_docs = {}

    for page in pages:
        parent_id = str(uuid.uuid4())
        page_text = page["text"]
        
        # 1. Store the Full Parent (The "Truth")
        parent_docs[parent_id] = page_text
        sub_texts = semantic_splitter.split_text(page_text)
        
        for text in sub_texts:
            child_doc = Document(
                page_content=text,
                metadata={
                    "parent_id": parent_id,
                    "source": page["metadata"].get("file_path"),
                    "page": page["metadata"].get("page_number")
                }
            )
            child_chunks.append(child_doc)
            
    return child_chunks, parent_docs