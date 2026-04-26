import os
import hashlib
from pathlib import Path
from dotenv import load_dotenv
from contextlib import contextmanager
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import sessionmaker, declarative_base
from urllib.parse import urlparse, urlencode, parse_qs, urlunparse
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, func

load_dotenv()

Base = declarative_base()


DOCUMENT_DIR = Path(__file__).resolve().parents[1] / "document_files"




class Document(Base):
    __tablename__ = "documents"

    id           = Column(Integer, primary_key=True)
    content_hash = Column(String, unique=True, nullable=False)
    source       = Column(String, nullable=False)
    created_at   = Column(DateTime, server_default=func.now())
    parent_id    = Column(UUID(as_uuid=True))
    parent_metadata = Column(JSONB, nullable = False)
    parent_content = Column(String, nullable=True)

    def __repr__(self):
        return f"<Document(id={self.id}, source='{self.source}, parent = {self.parent_id}')>"


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
        raw_url = db_url or os.getenv("DATABASE_URL")
        if not raw_url:
            raise ValueError("DATABASE_URL is not set")

        url = self._strip_param(raw_url, "channel_binding")

        self.engine = create_engine(
            url,
            pool_pre_ping=True,
            pool_recycle=300,
            pool_size=5,
            max_overflow=10,
        )
        Base.metadata.create_all(self.engine)

    @staticmethod
    def _strip_param(url: str, param: str) -> str:
        parsed = urlparse(url)
        params = parse_qs(parsed.query, keep_blank_values=True)
        params.pop(param, None)
        new_query = urlencode({k: v[0] for k, v in params.items()})
        return urlunparse(parsed._replace(query=new_query))

    @contextmanager
    def _session(self):
        Session = sessionmaker(bind=self.engine)
        session = Session()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()


    @staticmethod
    def _file_hash(filename: str) -> str:
        path = DOCUMENT_DIR / filename
        with open(path, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()

    def add_document(self, source: str, parent_id: str = None, parent_metadata: dict = None, parent_content: str = None) -> str:
        with self._session() as session:
            content_hash = self._file_hash(source)
            if session.query(Document).filter_by(content_hash=content_hash).first():
                return "Document already exists."
            session.add(Document(content_hash=content_hash, source=source, parent_id=parent_id, parent_metadata=parent_metadata, parent_content=parent_content))
        return "Document added successfully."

    def list_documents(self) -> list:
        with self._session() as session:
            docs = session.query(Document).order_by(Document.created_at.desc()).all()
            return [
                {"id": d.id, "source": d.source, "created_at": str(d.created_at)}
                for d in docs
            ]

    def get_document(self, doc_id: int):
        with self._session() as session:
            d = session.query(Document).filter_by(id=doc_id).first()
            if not d:
                return None
            return {"id": d.id, "source": d.source, "created_at": str(d.created_at)}

    def delete_document(self, doc_id: int) -> bool:
        with self._session() as session:
            doc = session.query(Document).filter_by(id=doc_id).first()
            if not doc:
                return False
            session.delete(doc)
        return True


    def add_message(self, conversation_id: str, role: str, content: str):
        with self._session() as session:
            session.add(
                Message(conversation_id=conversation_id, role=role, content=content)
            )

    def get_conversation(self, conversation_id: str, limit: int = 20) -> list:
        with self._session() as session:
            msgs = (
                session.query(Message)
                .filter_by(conversation_id=conversation_id)
                .order_by(Message.created_at.asc())
                .limit(limit)
                .all()
            )
            return [{"role": m.role, "content": m.content} for m in msgs]
