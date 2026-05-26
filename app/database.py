import os
from datetime import datetime, timezone

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    Index,
    Integer,
    String,
    Text,
    create_engine,
)
from sqlalchemy.orm import declarative_base, sessionmaker

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./rag_sentinel.db")

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {},
    pool_pre_ping=True,
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class PredictionLog(Base):
    __tablename__ = "prediction_logs"
    id = Column(Integer, primary_key=True, index=True)
    query = Column(Text, nullable=False)
    anomaly_score = Column(Float, nullable=False)
    is_anomaly = Column(Boolean, nullable=False, index=True)
    rag_context_used = Column(Boolean, default=False)
    response_time_ms = Column(Float)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), index=True)

    __table_args__ = (
        Index("ix_prediction_logs_created_is_anomaly", "created_at", "is_anomaly"),
    )

    def __repr__(self) -> str:
        return f"<PredictionLog id={self.id} score={self.anomaly_score} anomaly={self.is_anomaly}>"


class DriftLog(Base):
    __tablename__ = "drift_logs"
    id = Column(Integer, primary_key=True, index=True)
    ks_statistic = Column(Float, nullable=False)
    p_value = Column(Float, nullable=False)
    drift_detected = Column(Boolean, nullable=False)
    sample_size = Column(Integer)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), index=True)

    def __repr__(self) -> str:
        return f"<DriftLog id={self.id} ks={self.ks_statistic} drift={self.drift_detected}>"


class DocumentIndex(Base):
    __tablename__ = "document_index"
    id = Column(Integer, primary_key=True, index=True)
    doc_id = Column(String(64), unique=True, index=True)
    filename = Column(String(256))
    chunk_count = Column(Integer, default=0)
    indexed_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    def __repr__(self) -> str:
        return f"<DocumentIndex doc_id={self.doc_id} chunks={self.chunk_count}>"


def init_db() -> None:
    """Create all database tables if they do not exist."""
    Base.metadata.create_all(bind=engine)


def get_db():
    """Yield a SQLAlchemy session and ensure it is closed after use."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
