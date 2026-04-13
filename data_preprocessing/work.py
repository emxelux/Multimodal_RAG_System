from ingest import clean_text, load_pdf, path
from chunking import chunk_document
from vector_db import VectorDB



text, source = load_pdf(f"{path}/The Story of Jesus.pdf")
text = clean_text(text, source)

chunks = chunk_document(text)


my_client = VectorDB()
my_client.initialize(chunks[10:20])
print(my_client.search("Where was Jesus born"))
