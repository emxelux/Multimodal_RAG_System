import hashlib
import os
from dotenv import load_dotenv
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, func
from sqlalchemy.orm import sessionmaker, declarative_base

load_dotenv()

Base = declarative_base()
FULL_PATH = "document_files"


class Document(Base):
    __tablename__ = "documents"

    id           = Column(Integer, primary_key=True)
    content_hash = Column(String, unique=True, nullable=False)
    source       = Column(String, nullable=False)
    created_at   = Column(DateTime, server_default=func.now())

    def __repr__(self):
        return f"<Document(id={self.id}, source='{self.source}')>"


class Message(Base):
    __tablename__ = "messages"

    id              = Column(Integer, primary_key=True)
    conversation_id = Column(String, nullable=False, index=True)
    role            = Column(String, nullable=False)   # "user" | "assistant"
    content         = Column(Text, nullable=False)
    created_at      = Column(DateTime, server_default=func.now())

    def __repr__(self):
        return f"<Message(conv={self.conversation_id}, role={self.role})>"


class Database:
    def __init__(self, db_url: str = None):
        url = db_url or os.getenv("DATABASE_URL")
        if not url:
            raise ValueError("DATABASE_URL is not set")
        self.engine = create_engine(
            url,
            pool_pre_ping=True,
            pool_recycle=300,
            pool_size=5,
            max_overflow=10,
        )
        Base.metadata.create_all(self.engine)

    def _session(self):
        return sessionmaker(bind=self.engine)()

    @staticmethod
    def _file_hash(filename: str) -> str:
        path = f"{FULL_PATH}/{filename}"
        with open(path, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()

    def add_document(self, source: str) -> str:
        session = self._session()
        try:
            content_hash = self._file_hash(source)
            if session.query(Document).filter_by(content_hash=content_hash).first():
                return "Document already exists."
            session.add(Document(content_hash=content_hash, source=source))
            session.commit()
            return "Document added successfully."
        finally:
            session.close()

    def list_documents(self) -> list:
        session = self._session()
        try:
            docs = session.query(Document).order_by(Document.created_at.desc()).all()
            return [
                {"id": d.id, "source": d.source, "created_at": str(d.created_at)}
                for d in docs
            ]
        finally:
            session.close()

    def get_document(self, doc_id: int):
        session = self._session()
        try:
            d = session.query(Document).filter_by(id=doc_id).first()
            if not d:
                return None
            return {"id": d.id, "source": d.source, "created_at": str(d.created_at)}
        finally:
            session.close()

    def delete_document(self, doc_id: int) -> bool:
        session = self._session()
        try:
            doc = session.query(Document).filter_by(id=doc_id).first()
            if not doc:
                return False
            session.delete(doc)
            session.commit()
            return True
        finally:
            session.close()



    def add_message(self, conversation_id: str, role: str, content: str):
        session = self._session()
        try:
            session.add(
                Message(conversation_id=conversation_id, role=role, content=content)
            )
            session.commit()
        finally:
            session.close()

    def get_conversation(self, conversation_id: str, limit: int = 20) -> list:
        session = self._session()
        try:
            msgs = (
                session.query(Message)
                .filter_by(conversation_id=conversation_id)
                .order_by(Message.created_at.asc())
                .limit(limit)
                .all()
            )
            return [{"role": m.role, "content": m.content} for m in msgs]
        finally:
            session.close()
