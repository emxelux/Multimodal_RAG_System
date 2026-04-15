import os
import hashlib
from dotenv import load_dotenv
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy import create_engine, Column, Integer, String


load_dotenv()

Base = declarative_base()
db_url = os.getenv("DATABASE_URL")
full_path = "document_files"

class Document(Base):
    __tablename__ = 'documents'

    id = Column(Integer, primary_key=True)
    content_hash = Column(String)
    source = Column(String)

    def __repr__(self):
        return f"<Document(id={self.id}, source='{self.source}')>"

class Database:
    def __init__(self, db_url=db_url):
        self.engine = create_engine(db_url,pool_pre_ping=True,
pool_recycle=300, pool_size = 5, max_overflow = 10)
        Base.metadata.create_all(self.engine)

    @staticmethod
    def file_hash(path):
        with open(f"{full_path}/{path}", "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()

    def add_document(self, source) -> str:
        Session = sessionmaker(bind=self.engine)
        session = Session()
        existing_doc = session.query(Document).filter_by(content_hash=self.file_hash(f"{source}")).first()
        if not existing_doc:
            new_doc = Document(
                content_hash=self.file_hash(f"{source}"),
                source=source
            )
            session.add(new_doc)
            session.commit()
            session.close()
            return "Document added successfully"
        return "Document already Exist."

    def get_document(self, doc_id):
        Session = sessionmaker(bind=self.engine)
        session = Session()
        doc = session.query(Document).filter_by(id=doc_id).first()
        session.close()
        return doc