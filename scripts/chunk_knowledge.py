import json
import re
from collections import Counter
from pathlib import Path
import sys

import numpy as np
from transformers import AutoTokenizer

# CONFIG
# Đường dẫn input/output
ROOT_DIR = Path(__file__).resolve().parents[1]
INPUT_PATH = ROOT_DIR / "data/processed/knowledge.jsonl"
OUTPUT_PATH = ROOT_DIR / "data/processed/chunks.jsonl"

sys.path.insert(0, str(ROOT_DIR))
from backend.app.core.config import settings

# Model embedding dùng cho semantic chunking
MODEL_NAME = settings.EMBEDDING_MODEL_NAME

# Kích thước chunk theo token
MAX_TOKENS = 400          # Chunk tối đa 400 token
MIN_TOKENS = 100          #  Tránh tạo chunk quá nhỏ khi semantic splitting
ATOMIC_MAX_TOKENS = 180   # Unit dài hơn mức này sẽ tách tiếp thành câu

# Cấu hình semantic boundary
BREAK_PERCENTILE = 80     # Chọn ~20% điểm đổi ngữ nghĩa mạnh nhất để cân nhắc cắt
CONTEXT_BUFFER = 1        # Khi embed 1 unit, lấy thêm 1 unit trước và sau làm context

# Nhận diện heading Markdown như ##, ###, ####
HEADING_RE = re.compile(r"^(#{2,4})\s+(.+?)\s*$")

# Lazy load, tránh load tokenizer/model ngay khi script khởi động
_tokenizer = None
_model = None


def get_tokenizer():
    """Load tokenizer BGE-M3."""
    global _tokenizer
    if _tokenizer is None:
        print(f"Loading tokenizer: {MODEL_NAME}")
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    return _tokenizer


def get_model():
    """Load BGE-M3 khi cần semantic split."""
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        print(f"Loading semantic model: {MODEL_NAME}")
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def count_tokens(text):
    """Đếm token bằng tokenizer của BGE-M3."""
    return len(get_tokenizer().encode(text, add_special_tokens=False))


def read_jsonl(path):
    """Đọc JSONL."""
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


def split_markdown_blocks(text):
    """Tách medical content theo heading Markdown."""
    blocks, stack = [], []
    current_path, current_lines = None, []

    def flush():
        nonlocal current_lines
        content = "\n".join(current_lines).strip()
        if content:
            blocks.append((current_path, content))
        current_lines = []

    for raw_line in text.splitlines():
        line = raw_line.strip()
        match = HEADING_RE.match(line)

        if match:
            flush()
            level, title = len(match.group(1)), match.group(2).strip()
            stack = [(old_level, old_title) for old_level, old_title in stack if old_level < level]
            stack.append((level, title))
            current_path = " > ".join(title for _, title in stack)
            continue

        if line:
            current_lines.append(line)

    flush()
    return blocks if blocks else [(None, text.strip())]


def hard_split(text):
    """Fallback khi một unit vẫn vượt MAX_TOKENS."""
    words = text.split()
    chunks, current = [], []

    for word in words:
        candidate = " ".join(current + [word])

        if current and count_tokens(candidate) > MAX_TOKENS:
            chunks.append(" ".join(current))
            current = [word]
        else:
            current.append(word)

    if current:
        chunks.append(" ".join(current))

    return chunks


def split_atomic_units(text):
    """Tách đoạn dài thành câu hoặc bullet nhỏ."""
    units = []

    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue

        if count_tokens(line) <= ATOMIC_MAX_TOKENS:
            units.append(line)
            continue

        sentences = re.split(r"(?<=[.!?])\s+", line)

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            if count_tokens(sentence) <= MAX_TOKENS:
                units.append(sentence)
            else:
                units.extend(hard_split(sentence))

    return units


def build_context_windows(units):
    """Ghép unit lân cận để tính semantic similarity ổn định hơn."""
    windows = []

    for i in range(len(units)):
        start = max(0, i - CONTEXT_BUFFER)
        end = min(len(units), i + CONTEXT_BUFFER + 1)
        windows.append(" ".join(units[start:end]))

    return windows


def merge_small_chunks(chunks):
    """Gộp chunk quá nhỏ nếu vẫn nằm trong giới hạn token."""
    if not chunks:
        return []

    result = []

    for chunk in chunks:
        if result and count_tokens(chunk) < MIN_TOKENS:
            merged = f"{result[-1]}\n{chunk}"

            if count_tokens(merged) <= MAX_TOKENS:
                result[-1] = merged
                continue

        result.append(chunk)

    return result


def semantic_split(text):
    """Tách đoạn dài theo thay đổi ngữ nghĩa."""
    text = text.strip()

    if count_tokens(text) <= MAX_TOKENS:
        return [text]

    units = split_atomic_units(text)

    if len(units) <= 1:
        return hard_split(text)

    windows = build_context_windows(units)
    model = get_model()

    embeddings = model.encode(
        windows,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )

    similarities = np.sum(embeddings[:-1] * embeddings[1:], axis=1)
    distances = 1 - similarities
    threshold = np.percentile(distances, BREAK_PERCENTILE)

    chunks = []
    current = [units[0]]
    current_tokens = count_tokens(units[0])

    for i in range(1, len(units)):
        unit = units[i]
        unit_tokens = count_tokens(unit)
        candidate_tokens = current_tokens + unit_tokens

        semantic_break = distances[i - 1] >= threshold and current_tokens >= MIN_TOKENS
        size_break = candidate_tokens > MAX_TOKENS

        if semantic_break or size_break:
            chunks.append("\n".join(current).strip())
            current = [unit]
            current_tokens = unit_tokens
        else:
            current.append(unit)
            current_tokens = candidate_tokens

    if current:
        chunks.append("\n".join(current).strip())

    return merge_small_chunks(chunks)


def chunk_record(record):
    """Chunk một normalized record."""
    blocks = split_markdown_blocks(record["content"]) if record["document_type"] == "medical" else [(None, record["content"])]
    chunks = []

    for subsection, block_content in blocks:
        for content in semantic_split(block_content):
            chunks.append({
                "document_type": record["document_type"],
                "scope_type": record["scope_type"],
                "scope_value": record["scope_value"],
                "section": record["section"],
                "subsection": subsection,
                "source": record["source"],
                "source_url": record["source_url"],
                "chunk_index": len(chunks),
                "token_count": count_tokens(content),
                "content": content,
            })

    return chunks


def validate_chunks(chunks):
    """Kiểm tra chunks trước khi ghi."""
    errors = []
    required = {
        "document_type", "scope_type", "scope_value", "section", "subsection",
        "source", "source_url", "chunk_index", "token_count", "content",
    }

    for i, chunk in enumerate(chunks, 1):
        missing = required - chunk.keys()
        if missing:
            errors.append(f"chunk {i}: thiếu field {sorted(missing)}")
            continue

        if not chunk["content"]:
            errors.append(f"chunk {i}: content rỗng")

        actual_tokens = count_tokens(chunk["content"])
        if actual_tokens != chunk["token_count"]:
            errors.append(f"chunk {i}: token_count không khớp")

        if actual_tokens > MAX_TOKENS:
            errors.append(f"chunk {i}: quá dài ({actual_tokens} tokens)")

    return errors


def print_stats(chunks):
    """In thống kê chunking."""
    document_counts = Counter(chunk["document_type"] for chunk in chunks)
    section_counts = Counter(chunk["section"] for chunk in chunks)
    lengths = [chunk["token_count"] for chunk in chunks]

    print("\nCHUNKING STATS")
    print(f"Total chunks   : {len(chunks)}")
    print(f"Species chunks : {document_counts['species']}")
    print(f"Medical chunks : {document_counts['medical']}")

    print("\nSections:")
    for section, count in sorted(section_counts.items()):
        print(f"- {section:<20} {count}")

    if lengths:
        print("\nChunk tokens:")
        print(f"- min    : {min(lengths)}")
        print(f"- median : {int(np.median(lengths))}")
        print(f"- mean   : {int(np.mean(lengths))}")
        print(f"- max    : {max(lengths)}")


def write_jsonl(records, path):
    """Ghi chunks ra JSONL."""
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def main():
    print("Loading normalized knowledge...")
    records = read_jsonl(INPUT_PATH)
    print(f"Normalized records: {len(records)}")

    chunks = []

    for i, record in enumerate(records, 1):
        chunks.extend(chunk_record(record))

        if i % 50 == 0 or i == len(records):
            print(f"Processed {i}/{len(records)} records")

    errors = validate_chunks(chunks)

    if errors:
        print("\nVALIDATION ERRORS")
        for error in errors:
            print(f"- {error}")
        raise ValueError("Chunking failed. Hãy sửa lỗi trước khi ghi output.")

    print("\nValidation: OK")
    write_jsonl(chunks, OUTPUT_PATH)
    print_stats(chunks)

    print("\nSmallest chunks:")
    for chunk in sorted(chunks, key=lambda x: x["token_count"])[:10]:
        scope = chunk["scope_value"] or "general"
        print(
            f"- {chunk['token_count']:>3} tokens | "
            f"{scope} | {chunk['section']} | "
            f"{chunk['subsection']} | "
            f"{chunk['content'][:100]}"
        )

    print("\nCHUNKING COMPLETE")
    print(f"Output: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()