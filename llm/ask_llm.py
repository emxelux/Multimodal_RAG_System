from data_preprocessing.ingest import clean_text, load_pdf, DATA_DIR
from data_preprocessing.chunking import chunk_document
from data_preprocessing.vector_db import VectorDB
from llm.llm_connection import LLM

llm = LLM()

path = DATA_DIR



text, source = load_pdf(f"{path}/about.pdf")
text = clean_text(text, source)

chunks = chunk_document(text)


my_client = VectorDB()
my_client.initialize(chunks)
query = "what are the four things to give attention to?"
result = my_client.search(query)
# print(llm.generate_response(prompt))
print(system_prompt)