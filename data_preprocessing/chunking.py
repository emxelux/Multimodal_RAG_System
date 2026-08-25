from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_core.documents import Document


headers_to_split_on = [
    ("#", "Header1"),
    ("##", "Header2"),
    ("###", "Header3"),
]
header_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)


text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,      
    chunk_overlap=100,    
    separators=["\n\n", "\n", " ", ""] 
)

def split_markdown_document(documents: list[Document]):
    final_chunks = []
    
    for doc in documents:
        header_splits = header_splitter.split_text(doc.page_content)
        
        for section in header_splits:
            section.metadata.update(doc.metadata)
            
            sub_chunks = text_splitter.split_documents([section])
            
            for chunk in sub_chunks:
                header_context = " > ".join([val for key, val in chunk.metadata.items() if "Header" in key])
                if header_context:
                    chunk.page_content = f"Context: {header_context}\n\n{chunk.page_content}"
                
                final_chunks.append(chunk)
                
    return final_chunks