import os
from dotenv import load_dotenv

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()


load_dotenv()


SQLALCHEMY_DATABASE_URL = os.getenv("DATABASE_URL")


engine = create_engine(
    SQLALCHEMY_DATABASE_URL,
    # 1. Checks if connection is alive before executing a query (pings the DB)
    pool_pre_ping=True, 
    # 2. Automatically recycles connections older than 1 hour (3600 seconds)
    pool_recycle=3600,  
    # 3. Keeps a stable pool of connections
    pool_size=10,
    max_overflow=20
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
