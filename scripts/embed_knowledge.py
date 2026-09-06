import csv
import json
import sys
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

# CONFIG
ROOT_DIR = Path(__file__).resolve().parents[1]
INPUT_PATH = ROOT_DIR / "data/processed/chunks.jsonl"
OUTPUT_PATH = ROOT_DIR / "data/processed/embeddings.npy"
SPECIES_METADATA_PATH = ROOT_DIR / "data/processed/species_metadata.csv"

# Cho phép import backend khi chạy trực tiếp script
sys.path.insert(0, str(ROOT_DIR))
from backend.app.core.config import settings

# Model embedding
MODEL_NAME = settings.EMBEDDING_MODEL_NAME
EMBEDDING_DIM = settings.EMBEDDING_DIM
BATCH_SIZE = 16

# Nhãn semantic dùng khi embedding
SECTION_LABELS = {
    "diagnosis": "Chẩn đoán, triệu chứng, dấu hiệu",
    "treatment": "Điều trị",
    "first_aid": "Sơ cứu",
    "identification": "Đặc điểm nhận dạng",
    "distribution": "Phân bố",
    "habitat_behavior": "Môi trường sống, tập tính",
    "venom_danger": "Nọc độc, mức độ nguy hiểm",
}


def read_jsonl(path):
    """Đọc chunks từ JSONL."""
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy: {path}")

    records = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path.name} dòng {line_no}: JSON lỗi - {exc}") from exc

    return records


def load_species_names(path):
    """Map tên khoa học sang tên tiếng Việt."""
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy: {path}")

    species_names = {}
    with path.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            binomial_name = row["binomial_name"].strip()
            vietnamese_name = row.get("vietnamese_name", "").strip()

            if vietnamese_name:
                species_names[binomial_name] = vietnamese_name

    return species_names


def build_embedding_text(chunk, species_names):
    """Tạo text có context để embedding."""
    scope = chunk["scope_value"]
    section = SECTION_LABELS.get(chunk["section"], chunk["section"])
    parts = []

    if chunk["scope_type"] == "species" and scope:
        parts.append(f"Species: {scope}")

        vietnamese_name = species_names.get(scope)
        if vietnamese_name:
            parts.append(f"Tên tiếng Việt: {vietnamese_name}")
    else:
        parts.append(f"Scope: {scope or 'Rắn cắn nói chung'}")

    parts.append(f"Section: {section}")

    if chunk.get("subsection"):
        parts.append(f"Subsection: {chunk['subsection']}")

    parts.extend(["", chunk["content"]])
    return "\n".join(parts)


def validate_chunks(chunks):
    """Kiểm tra input trước khi embedding."""
    errors = []
    required = {"scope_type", "scope_value", "section", "subsection", "content"}

    for i, chunk in enumerate(chunks, 1):
        missing = required - chunk.keys()
        if missing:
            errors.append(f"chunk {i}: thiếu field {sorted(missing)}")
            continue

        if not chunk["content"].strip():
            errors.append(f"chunk {i}: content rỗng")

    return errors


def embed_chunks(chunks, species_names):
    """Embedding toàn bộ chunks bằng BGE-M3."""
    print(f"Loading embedding model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)

    texts = [build_embedding_text(chunk, species_names) for chunk in chunks]
    print(f"Embedding {len(texts)} chunks...")

    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True,
    )

    return embeddings.astype(np.float32)


def validate_embeddings(chunks, embeddings):
    """Kiểm tra output embedding."""
    if len(embeddings) != len(chunks):
        raise ValueError(f"Số embeddings ({len(embeddings)}) không khớp số chunks ({len(chunks)})")

    if embeddings.ndim != 2:
        raise ValueError(f"Embedding phải là ma trận 2D, nhận được shape {embeddings.shape}")

    if embeddings.shape[1] != EMBEDDING_DIM:
        raise ValueError(f"Embedding dimension phải là {EMBEDDING_DIM}, nhận được {embeddings.shape[1]}")

    if not np.isfinite(embeddings).all():
        raise ValueError("Embedding chứa NaN hoặc Infinity")


def main():
    print("Loading chunks...")
    chunks = read_jsonl(INPUT_PATH)
    species_names = load_species_names(SPECIES_METADATA_PATH)

    print(f"Chunks        : {len(chunks)}")
    print(f"Species names : {len(species_names)}")

    errors = validate_chunks(chunks)
    if errors:
        print("\nVALIDATION ERRORS")
        for error in errors:
            print(f"- {error}")
        raise ValueError("Embedding failed. Hãy sửa chunks trước.")

    print("Validation: OK")

    embeddings = embed_chunks(chunks, species_names)
    validate_embeddings(chunks, embeddings)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.save(OUTPUT_PATH, embeddings)

    print("\nEMBEDDING STATS")
    print(f"Chunks     : {len(chunks)}")
    print(f"Shape      : {embeddings.shape}")
    print(f"Dimension  : {embeddings.shape[1]}")
    print(f"Dtype      : {embeddings.dtype}")
    print("Normalized : True")
    print(f"\nOutput: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()