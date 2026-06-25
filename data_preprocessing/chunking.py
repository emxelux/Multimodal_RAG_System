# from langchain_text_splitters.markdown import MarkdownHeaderTextSplitter
# from langchain_core.documents import Document

# headers_to_split_on = [
#     ("#", "Header1"),
#     ("##", "Header2"),
#     ("###", "Header3"),
# ]
# header_splitter = MarkdownHeaderTextSplitter(
#     headers_to_split_on=headers_to_split_on
# )


# def split_markdown_document(documents:Document):
#     header_docs = []
#     for doc in documents:
#         splits = header_splitter.split_text(
#             doc.page_content
#         )
#         for split in splits:
#             split.metadata.update(doc.metadata)
#             header_docs.append(split)
#     return header_docs

from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# Step 1: Define header splitting
headers_to_split_on = [
    ("#", "Header1"),
    ("##", "Header2"),
    ("###", "Header3"),
]
header_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)

# Step 2: Define text splitting for oversized sections
# We use specific markdown characters in the separators to avoid cutting tables mid-row
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,       # Adjust based on your LLM's context window comfort
    chunk_overlap=100,     # Keeps context continuity between splits
    separators=["\n\n", "\n", " ", ""] 
)

def split_markdown_document(documents: list[Document]):
    final_chunks = []
    
    for doc in documents:
        # First pass: Split into structural sections based on headers
        header_splits = header_splitter.split_text(doc.page_content)
        
        for section in header_splits:
            # Carry over original document metadata (e.g., source, author)
            section.metadata.update(doc.metadata)
            
            # Second pass: If a section is too large, split it recursively
            sub_chunks = text_splitter.split_documents([section])
            
            # Inject header context back into the text body so the chunk is self-contained
            for chunk in sub_chunks:
                header_context = " > ".join([val for key, val in chunk.metadata.items() if "Header" in key])
                if header_context:
                    # Prepend the context directly into the text
                    chunk.page_content = f"Context: {header_context}\n\n{chunk.page_content}"
                
                final_chunks.append(chunk)
                
    return final_chunks