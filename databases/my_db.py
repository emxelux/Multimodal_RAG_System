from databases.database import Database, Document
from data_preprocessing.ingest import path

source = f"{path}/The Story of Jesus.pdf"

db = Database()

print(db.add_document(source=source))

