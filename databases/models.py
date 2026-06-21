from sqlalchemy import Column, Integer, String, Boolean, ForeignKey
from sqlalchemy.orm import relationship
from sqlalchemy.sql.expression import text
# from uuid import UUID, uuid4
from sqlmodel import UUID
from sqlalchemy.sql.sqltypes import TIMESTAMP

from .database import Base


class User(Base):
    __tablename__ = "users"

    id = Column(UUID, primary_key=True, nullable=False)
    first_name = Column(String, nullable=False)
    last_name = Column(String, nullable=False)
    email = Column(String, nullable=False, unique=True)
    phone_number = Column(String, nullable=True)
    password = Column(String, nullable=False)
    is_verified = Column(Boolean, server_default='FALSE', nullable=False)
    created_at = Column(TIMESTAMP(timezone=True),
                        nullable=False, server_default=text('now()'))


class Document(Base):
    __tablename__ = "documents"

    id = Column(UUID, primary_key=True, nullable=False)
    document_hash = Column(String, nullable=False)
    uploaded_time = Column(TIMESTAMP(timezone=True),
                           nullable=False, server_default=text('now()'))
    user_id = Column(UUID, ForeignKey(
        "users.id", ondelete="CASCADE"), nullable=False)

    # owner = relationship("User")