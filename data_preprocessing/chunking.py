from langchain_text_splitters.markdown import MarkdownHeaderTextSplitter
from langchain_core.documents import Document

headers_to_split_on = [
    ("#", "Header1"),
    ("##", "Header2"),
    ("###", "Header3"),
]
header_splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=headers_to_split_on
)


def split_markdown_document(documents:Document):
    header_docs = []
    for doc in documents:
        splits = header_splitter.split_text(
            doc.page_content
        )
        for split in splits:
            split.metadata.update(doc.metadata)
            header_docs.append(split)
    return header_docs