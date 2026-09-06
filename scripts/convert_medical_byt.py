import json
import re
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]

INPUT_PATH = ROOT_DIR / "data/raw/medical_byt.txt"
OUTPUT_PATH = ROOT_DIR / "data/raw/medical_byt.jsonl"

SOURCE = "Bo Y Te"
SOURCE_URL = "https://kcb.vn/van-ban/huong-dan-chan-doan-va-xu-tri-ngo-doc.html"

SCOPE_MAP = {
    "RẮN HỔ MANG": {
        "scope_type": "group",
        "scope_value": "Naja atra; Naja kaouthia",
    },
    "RẮN HỔ MÈO": {
        "scope_type": "species",
        "scope_value": "Naja siamensis",
    },
    "RẮN CẠP NIA": {
        "scope_type": "genus",
        "scope_value": "Bungarus",
    },
    "RẮN LỤC": {
        "scope_type": "family",
        "scope_value": "Viperidae",
    },
    "RẮN CHÀM QUẠP": {
        "scope_type": "species",
        "scope_value": "Calloselasma rhodostoma",
    },
}

GENERAL_HEADING = "SƠ CỨU CHUNG KHI KHÔNG XÁC ĐỊNH ĐƯỢC LOẠI RẮN"

TOP_HEADING_RE = re.compile(r"^===\s*(.*?)\s*===\s*$")
SECTION_RE = re.compile(r"^##\s*(.*?)\s*$")


def clean_content(lines):
    """Làm sạch khoảng trắng nhưng giữ cấu trúc markdown."""
    while lines and not lines[0].strip():
        lines.pop(0)

    while lines and not lines[-1].strip():
        lines.pop()

    cleaned = []
    previous_blank = False

    for line in lines:
        line = line.rstrip()

        if not line.strip():
            if not previous_blank:
                cleaned.append("")
            previous_blank = True
            continue

        cleaned.append(line)
        previous_blank = False

    return "\n".join(cleaned).strip()


def split_blocks(text):
    """Tách file theo các heading === LOẠI RẮN ===."""
    blocks = {}
    current_name = None
    current_lines = []

    for line in text.splitlines():
        match = TOP_HEADING_RE.match(line.strip())

        if match:
            if current_name:
                blocks[current_name] = clean_content(current_lines)

            current_name = match.group(1).strip()
            current_lines = []
            continue

        if current_name:
            current_lines.append(line)

    if current_name:
        blocks[current_name] = clean_content(current_lines)

    return blocks


def split_sections(block):
    """Tách ## CHẨN ĐOÁN và ## ĐIỀU TRỊ."""
    sections = {}
    current_section = None
    current_lines = []

    for line in block.splitlines():
        match = SECTION_RE.match(line.strip())

        if match:
            name = match.group(1).strip().upper()

            if name in {"CHẨN ĐOÁN", "ĐIỀU TRỊ"}:
                if current_section:
                    sections[current_section] = clean_content(current_lines)

                current_section = name
                current_lines = []
                continue

        if current_section:
            current_lines.append(line)

    if current_section:
        sections[current_section] = clean_content(current_lines)

    return sections


def make_record(scope_type, scope_value, section, content, source=SOURCE):
    """Tạo một raw medical record."""
    return {
        "scope_type": scope_type,
        "scope_value": scope_value,
        "section": section,
        "source": source,
        "source_url": SOURCE_URL,
        "content": content,
    }


def collect_snake_records(blocks):
    """Tạo diagnosis và treatment cho từng nhóm rắn."""
    records = []

    for heading, scope in SCOPE_MAP.items():
        block = blocks.get(heading)

        if not block:
            print(f"WARNING: Không tìm thấy block {heading}")
            continue

        sections = split_sections(block)

        diagnosis = sections.get("CHẨN ĐOÁN")
        treatment = sections.get("ĐIỀU TRỊ")

        if diagnosis:
            records.append(
                make_record(
                    scope["scope_type"],
                    scope["scope_value"],
                    "diagnosis",
                    diagnosis,
                )
            )
        else:
            print(f"WARNING: {heading} thiếu CHẨN ĐOÁN")

        if treatment:
            records.append(
                make_record(
                    scope["scope_type"],
                    scope["scope_value"],
                    "treatment",
                    treatment,
                )
            )
        else:
            print(f"WARNING: {heading} thiếu ĐIỀU TRỊ")

    return records


def collect_general_first_aid(blocks):
    """Thu thập phần sơ cứu chung do mình tổng hợp từ BYT."""
    block = blocks.get(GENERAL_HEADING)

    if not block:
        print("WARNING: Không tìm thấy phần sơ cứu chung")
        return None

    return make_record(
        scope_type="general",
        scope_value=None,
        section="first_aid",
        content=block,
        source="Bo Y Te - Tong hop",
    )


def remove_duplicates(records):
    """Loại record trùng hoàn toàn."""
    seen = set()
    result = []

    for record in records:
        key = (
            record["scope_type"],
            record["scope_value"],
            record["section"],
            record["content"],
        )

        if key not in seen:
            seen.add(key)
            result.append(record)

    return result


def validate_records(records):
    """Kiểm tra JSONL trước khi ghi."""
    errors = []

    required_fields = {
        "scope_type",
        "scope_value",
        "section",
        "source",
        "source_url",
        "content",
    }

    for i, record in enumerate(records, 1):
        missing = required_fields - record.keys()

        if missing:
            errors.append(f"record {i}: missing {sorted(missing)}")

        if not record["content"].strip():
            errors.append(f"record {i}: empty content")

        if record["section"] not in {"diagnosis", "treatment", "first_aid"}:
            errors.append(
                f"record {i}: invalid section {record['section']}"
            )

    expected = 11

    if len(records) != expected:
        errors.append(
            f"expected {expected} records, got {len(records)}"
        )

    if errors:
        print("\nVALIDATION WARNINGS")
        for error in errors:
            print(f"- {error}")
        return False

    print("\nValidation: OK")
    return True


def main():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Không tìm thấy: {INPUT_PATH}")

    text = INPUT_PATH.read_text(encoding="utf-8")

    blocks = split_blocks(text)

    print("Detected blocks:")
    for name in blocks:
        print(f"- {name}")

    records = collect_snake_records(blocks)

    general_record = collect_general_first_aid(blocks)
    if general_record:
        records.append(general_record)

    records = remove_duplicates(records)
    validate_records(records)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(
                json.dumps(record, ensure_ascii=False) + "\n"
            )

    print("\nCONVERSION COMPLETE")
    print(f"Records : {len(records)}")
    print(f"Output  : {OUTPUT_PATH}")

    for record in records:
        scope = record["scope_value"] or "general"
        print(
            f"- {record['scope_type']}:{scope}"
            f" | {record['section']}"
            f" | {len(record['content'])} chars"
        )


if __name__ == "__main__":
    main()