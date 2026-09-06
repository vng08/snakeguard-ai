import json
import sys
from pathlib import Path

import numpy as np
from sqlalchemy import func, select

# CONFIG
ROOT_DIR = Path(__file__).resolve().parents[1]
CHUNKS_PATH = ROOT_DIR / "data/processed/chunks.jsonl"
EMBEDDINGS_PATH = ROOT_DIR / "data/processed/embeddings.npy"

# Cho phép import backend khi chạy trực tiếp script
sys.path.insert(0, str(ROOT_DIR))

from backend.app.db.models import KnowledgeDocument, SnakeSpecies
from backend.app.db.session import SessionLocal


def read_jsonl(path):
    """Đọc chunks từ JSONL."""
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_species_map(db):
    """Map binomial_name sang species_id."""
    rows = db.execute(select(SnakeSpecies.id, SnakeSpecies.binomial_name)).all()
    return {binomial_name: species_id for species_id, binomial_name in rows}


def build_document(chunk, embedding, species_map):
    """Tạo KnowledgeDocument từ chunk và embedding."""
    species_id = None

    if chunk["document_type"] == "species" and chunk["scope_type"] == "species":
        species_id = species_map.get(chunk["scope_value"])
        if species_id is None:
            raise ValueError(f"Không tìm thấy species trong DB: {chunk['scope_value']}")

    return KnowledgeDocument(
        species_id=species_id,
        document_type=chunk["document_type"],
        scope_type=chunk["scope_type"],
        scope_value=chunk["scope_value"],
        section=chunk["section"],
        subsection=chunk.get("subsection"),
        chunk_index=chunk["chunk_index"],
        content=chunk["content"],
        source=chunk["source"],
        source_url=chunk.get("source_url"),
        embedding=embedding.tolist(),
    )


def ingest(chunks, embeddings):
    """Insert knowledge vào PostgreSQL."""
    db = SessionLocal()

    try:
        existing = db.scalar(select(func.count()).select_from(KnowledgeDocument))
        if existing:
            raise ValueError(f"knowledge_documents đã có {existing} records, dừng để tránh duplicate.")

        species_map = load_species_map(db)
        print(f"Species trong DB: {len(species_map)}")

        documents = [
            build_document(chunk, embedding, species_map)
            for chunk, embedding in zip(chunks, embeddings)
        ]

        db.add_all(documents)
        db.commit()
        return len(documents)

    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def main():
    print("Loading chunks và embeddings...")
    chunks = read_jsonl(CHUNKS_PATH)
    embeddings = np.load(EMBEDDINGS_PATH)

    print(f"Chunks     : {len(chunks)}")
    print(f"Embeddings : {embeddings.shape}")

    inserted = ingest(chunks, embeddings)
    print(f"\nINGESTION COMPLETE\nInserted: {inserted} documents")


if __name__ == "__main__":
    main()