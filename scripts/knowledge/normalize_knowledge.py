import json
import re
from collections import Counter
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]

SPECIES_PATH = ROOT_DIR / "data/raw/species_knowledge.jsonl"
MEDICAL_PATH = ROOT_DIR / "data/raw/medical_byt.jsonl"
OUTPUT_PATH = ROOT_DIR / "data/processed/knowledge.jsonl"

EXPECTED_SPECIES = 109

SPECIES_SECTIONS = {"identification", "distribution", "habitat_behavior", "venom_danger"}
MEDICAL_SECTIONS = {"diagnosis", "treatment", "first_aid"}
VALID_SCOPE_TYPES = {"species", "genus", "family", "group", "general"}


def read_jsonl(path):
    """Đọc file JSONL."""
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


def clean_species_content(text):
    """Chuẩn hóa khoảng trắng species knowledge."""
    if not isinstance(text, str):
        return ""

    text = text.replace("\u00ad", "").replace("\ufeff", "")
    return re.sub(r"\s+", " ", text).strip()


def clean_medical_content(text):
    """Làm sạch medical text nhưng giữ markdown heading."""
    if not isinstance(text, str):
        return ""

    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\u00ad", "").replace("\ufeff", "")

    lines = []
    for line in text.splitlines():
        line = re.sub(r"[ \t]+", " ", line).strip()

        if line:
            lines.append(line)
        elif lines and lines[-1] != "":
            lines.append("")

    while lines and lines[0] == "":
        lines.pop(0)
    while lines and lines[-1] == "":
        lines.pop()

    return "\n".join(lines).strip()


def normalize_species_record(record):
    """Đưa species raw về schema chung."""
    return {
        "document_type": "species",
        "scope_type": "species",
        "scope_value": str(record.get("binomial_name", "")).strip(),
        "section": str(record.get("section", "")).strip(),
        "source": str(record.get("source", "")).strip(),
        "source_url": str(record.get("source_url", "")).strip(),
        "content": clean_species_content(record.get("content")),
    }


def normalize_medical_record(record):
    """Đưa medical raw về schema chung."""
    scope_value = record.get("scope_value")
    if isinstance(scope_value, str):
        scope_value = scope_value.strip() or None

    return {
        "document_type": "medical",
        "scope_type": str(record.get("scope_type", "")).strip(),
        "scope_value": scope_value,
        "section": str(record.get("section", "")).strip(),
        "source": str(record.get("source", "")).strip(),
        "source_url": str(record.get("source_url", "")).strip(),
        "content": clean_medical_content(record.get("content")),
    }


def remove_exact_duplicates(records):
    """Loại record trùng hoàn toàn."""
    seen, result = set(), []

    for record in records:
        key = (
            record["document_type"], record["scope_type"], record["scope_value"],
            record["section"], record["source"], record["source_url"], record["content"],
        )

        if key not in seen:
            seen.add(key)
            result.append(record)

    return result


def validate_records(records):
    """Kiểm tra schema normalized."""
    errors, warnings = [], []
    required_fields = {
        "document_type", "scope_type", "scope_value",
        "section", "source", "source_url", "content",
    }

    for i, record in enumerate(records, 1):
        missing = required_fields - record.keys()
        if missing:
            errors.append(f"record {i}: thiếu field {sorted(missing)}")
            continue

        document_type = record["document_type"]
        scope_type = record["scope_type"]
        section = record["section"]

        if document_type not in {"species", "medical"}:
            errors.append(f"record {i}: document_type không hợp lệ {document_type!r}")

        if scope_type not in VALID_SCOPE_TYPES:
            errors.append(f"record {i}: scope_type không hợp lệ {scope_type!r}")

        if scope_type != "general" and not record["scope_value"]:
            errors.append(f"record {i}: scope_value rỗng")

        if scope_type == "general" and record["scope_value"] is not None:
            warnings.append(f"record {i}: general scope có scope_value")

        if document_type == "species":
            if scope_type != "species":
                errors.append(f"record {i}: species document phải có scope_type=species")
            if section not in SPECIES_SECTIONS:
                errors.append(f"record {i}: species section không hợp lệ {section!r}")

        if document_type == "medical" and section not in MEDICAL_SECTIONS:
            errors.append(f"record {i}: medical section không hợp lệ {section!r}")

        if not record["source"]:
            errors.append(f"record {i}: source rỗng")
        if not record["source_url"]:
            errors.append(f"record {i}: source_url rỗng")
        if not record["content"]:
            errors.append(f"record {i}: content rỗng")

    return errors, warnings


def validate_species_coverage(records):
    """Kiểm tra đủ 109 species."""
    species = {
        record["scope_value"]
        for record in records
        if record["document_type"] == "species" and record["scope_value"]
    }

    if len(species) != EXPECTED_SPECIES:
        return [f"Expected {EXPECTED_SPECIES} species, found {len(species)}"]

    return []


def print_stats(records):
    """In thống kê sau normalize."""
    document_counts = Counter(record["document_type"] for record in records)
    species_sections = Counter(record["section"] for record in records if record["document_type"] == "species")
    medical_sections = Counter(record["section"] for record in records if record["document_type"] == "medical")
    sources = Counter(record["source"] for record in records)
    unique_species = {
        record["scope_value"]
        for record in records
        if record["document_type"] == "species"
    }

    print("\nNORMALIZATION STATS")
    print(f"Total records   : {len(records)}")
    print(f"Species records : {document_counts['species']}")
    print(f"Medical records : {document_counts['medical']}")
    print(f"Unique species  : {len(unique_species)}")

    print("\nSpecies sections:")
    for section, count in sorted(species_sections.items()):
        print(f"- {section:<20} {count}")

    print("\nMedical sections:")
    for section, count in sorted(medical_sections.items()):
        print(f"- {section:<20} {count}")

    print("\nSources:")
    for source, count in sorted(sources.items()):
        print(f"- {source:<25} {count}")


def write_jsonl(records, path):
    """Ghi dữ liệu normalized ra JSONL."""
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def main():
    print("Loading raw knowledge...")

    species_raw = read_jsonl(SPECIES_PATH)
    medical_raw = read_jsonl(MEDICAL_PATH)

    print(f"Species raw : {len(species_raw)}")
    print(f"Medical raw : {len(medical_raw)}")

    species_records = [normalize_species_record(record) for record in species_raw]
    medical_records = [normalize_medical_record(record) for record in medical_raw]
    records = species_records + medical_records

    before = len(records)
    records = remove_exact_duplicates(records)
    duplicates = before - len(records)

    errors, warnings = validate_records(records)
    warnings.extend(validate_species_coverage(records))

    if warnings:
        print("\nVALIDATION WARNINGS")
        for warning in warnings:
            print(f"- {warning}")

    if errors:
        print("\nVALIDATION ERRORS")
        for error in errors:
            print(f"- {error}")
        raise ValueError("Normalization failed. Hãy sửa lỗi trước khi ghi output.")

    print("\nValidation: OK")

    if duplicates:
        print(f"Exact duplicates removed: {duplicates}")

    write_jsonl(records, OUTPUT_PATH)
    print_stats(records)

    print("\nNORMALIZATION COMPLETE")
    print(f"Output: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()