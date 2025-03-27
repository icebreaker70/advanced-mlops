import os

from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker


load_dotenv()

feature_store_url = os.getenv("FEATURE_STORE_URL")

engine = create_engine(feature_store_url, echo=False)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)



Base = declarative_base()


def get_db():
    """데이터베이스 세션을 제공하는 함수 (의존성 주입용)"""
    with SessionLocal() as session:
        yield session
