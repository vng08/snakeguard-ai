from datetime import datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    JSON,
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
)

from backend.app.db.base import Base


class SnakeSpecies(Base):
    __tablename__ = "snake_species"

    id = Column(Integer, primary_key=True, autoincrement=True)

    binomial_name = Column(String(255), unique=True, nullable=False)
    vietnamese_name = Column(String(255), nullable=True)

    family = Column(String(100), nullable=True)
    genus = Column(String(100), nullable=True)

    is_mivs = Column(Boolean, nullable=False)

    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False,)


class SnakeImage(Base):
    __tablename__ = "snake_images"

    id = Column(Integer, primary_key=True, autoincrement=True)
    species_id = Column(Integer, ForeignKey("snake_species.id"), nullable=False)

    image_url = Column(String(500), nullable=False)
    source = Column(String(255), nullable=True)
    license = Column(String(100), nullable=True)

    embedding = Column(Vector(768), nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)


class PredictionLog(Base):
    __tablename__ = "prediction_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    image_url = Column(String(500), nullable=True)
    bbox = Column(JSON, nullable=True)
    detection_confidence = Column(Float, nullable=True)

    predicted_species_id = Column(Integer, ForeignKey("snake_species.id"), nullable=True)
    confidence = Column(Float, nullable=True)
    top_k_predictions = Column(JSON, nullable=True)
    model_version = Column(String(100), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

class KnowledgeDocument(Base):
    __tablename__ = "knowledge_documents"

    id = Column(Integer, primary_key=True, autoincrement=True)
    species_id = Column(Integer, ForeignKey("snake_species.id"), nullable=True)

    document_type = Column(String(50), nullable=False)
    scope_type = Column(String(50), nullable=False)
    scope_value = Column(String(255), nullable=True)

    section = Column(String(100), nullable=False)
    subsection = Column(String(255), nullable=True)
    chunk_index = Column(Integer, nullable=False)

    content = Column(Text, nullable=False)
    source = Column(String(255), nullable=False)
    source_url = Column(String(500), nullable=True)

    embedding = Column(Vector(1024), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

class ChatMessage(Base):
    __tablename__ = "chat_messages"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(36), nullable=False, index=True)
    role = Column(String(20), nullable=False)
    content = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)