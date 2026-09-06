import csv
import json
import re
import time
from pathlib import Path

import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

ROOT_DIR = Path(__file__).resolve().parents[1]

# Đường dẫn input/output
MISSING_CSV = ROOT_DIR / "data/reports/vietnamsnakes_missing.csv"
OUTPUT_JSONL = ROOT_DIR / "data/raw/species_knowledge.jsonl"
REPORT_CSV = ROOT_DIR / "data/reports/missing_collection_report.csv"

# Cấu hình The Reptile Database
BASE_URL = "https://reptile-database.reptarium.cz"
SOURCE = "The Reptile Database"
REQUEST_DELAY = 1

def clean_text(text):
    """Chuẩn hóa khoảng trắng và ký tự thừa."""
    text = text.replace("\u00ad", "")
    return re.sub(r"\s+", " ", text).strip(" \n\t;")

def create_session():
    """Tạo HTTP session có retry khi gặp lỗi mạng/server."""
    session = requests.Session()
    retry = Retry(total=3, connect=3, read=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504], allowed_methods=["GET"])
    session.mount("https://", HTTPAdapter(max_retries=retry))
    session.headers.update({"User-Agent": "SnakeGuardAI/1.0 educational research project"})
    return session

def load_missing_species():
    """Đọc danh sách species chưa tìm thấy trên VietnamSnakes."""
    with MISSING_CSV.open(encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        return []

    possible_columns = ["binomial_name", "species", "scientific_name"]
    column = next((c for c in possible_columns if c in rows[0]), None)
    if not column:
        raise ValueError(f"Cannot find species column. Columns: {list(rows[0])}")

    return [clean_text(row[column]) for row in rows if clean_text(row.get(column, ""))]

def species_url(binomial_name):
    """Tạo URL species từ tên khoa học."""
    parts = binomial_name.split()
    if len(parts) < 2:
        return None
    return f"{BASE_URL}/{parts[0]}/{parts[1]}"

def parse_fields(soup):
    """Đọc các field dạng bảng trên trang species."""
    fields = {}

    for row in soup.find_all("tr"):
        cells = row.find_all(["th", "td"])
        if len(cells) < 2:
            continue

        key = clean_text(cells[0].get_text(" ", strip=True)).rstrip(":")
        value = clean_text(cells[1].get_text(" ", strip=True))
        if key and value:
            fields[key] = value

    return fields

def valid_species_page(soup, binomial_name):
    """Kiểm tra trang trả về có đúng species cần tìm."""
    h1 = soup.find("h1")
    if not h1:
        return False

    title = clean_text(h1.get_text(" ", strip=True)).lower()
    genus, species = binomial_name.lower().split()[:2]
    return genus in title and species in title

def clean_distribution(text):
    """Loại Type locality khỏi thông tin phân bố."""
    text = re.split(r"\bType locality\s*:", text, maxsplit=1, flags=re.I)[0]
    return clean_text(text)

def clean_diagnosis(text):
    """Làm sạch phần đặc điểm nhận dạng."""
    text = re.sub(r"^Diagnosis\s*:\s*", "", text, flags=re.I)
    boilerplate = "Unfortunately we had to temporarily remove additional information"

    if boilerplate.lower() in text.lower():
        text = text[:text.lower().find(boilerplate.lower())]

    return clean_text(text)

def extract_labeled_text(text, labels):
    """Lấy Habitat/Ecology/Behavior nằm bên trong Comment."""
    parts = []
    stop_labels = "Habitat|Ecology|Behavior|Behaviour|Diet|Distribution|Abundance|Variation|Synonymy|Taxonomy|Subspecies|Type species|Etymology|Reproduction|Genome"

    for label in labels:
        match = re.search(rf"\b{re.escape(label)}\s*:\s*(.*?)(?=\b(?:{stop_labels})\s*:|$)", text, flags=re.I)
        if not match:
            continue

        value = clean_text(match.group(1))
        if value:
            parts.append(f"{label}: {value}")

    return clean_text(" ".join(parts))

def extract_venom_info(comment):
    """Lấy các câu liên quan trực tiếp đến độc tính/nguy hiểm."""
    if not comment:
        return ""

    sentences = re.split(r"(?<=[.!?])\s+", comment)
    keywords = ["venom", "venomous", "toxic", "dangerous", "poison"]
    selected = [clean_text(s) for s in sentences if any(k in s.lower() for k in keywords)]
    return clean_text(" ".join(selected))

def make_record(binomial_name, section, url, content):
    """Tạo một raw knowledge record."""
    return {
        "binomial_name": binomial_name,
        "section": section,
        "source": SOURCE,
        "source_url": url,
        "content": content,
    }

def collect_species(session, binomial_name):
    """Thu thập knowledge của một species từ The Reptile Database."""
    url = species_url(binomial_name)
    if not url:
        return [], "invalid_name", None

    response = session.get(url, timeout=(10, 60))
    if response.status_code != 200:
        return [], f"http_{response.status_code}", url

    soup = BeautifulSoup(response.text, "html.parser")
    if not valid_species_page(soup, binomial_name):
        return [], "species_not_found", url

    fields = parse_fields(soup)
    records = []

    # Phân bố
    distribution = clean_distribution(fields.get("Distribution", ""))
    if distribution:
        records.append(make_record(binomial_name, "distribution", url, distribution))

    # Đặc điểm nhận dạng
    diagnosis = clean_diagnosis(fields.get("Diagnosis", ""))
    if len(diagnosis) >= 30:
        records.append(make_record(binomial_name, "identification", url, diagnosis))

    comment = fields.get("Comment", "")

    # Chỉ tạo habitat_behavior khi thật sự có Habitat/Ecology/Behavior
    habitat_behavior = extract_labeled_text(comment, ["Habitat", "Ecology", "Behavior", "Behaviour"])
    if habitat_behavior:
        records.append(make_record(binomial_name, "habitat_behavior", url, habitat_behavior))

    # Thông tin độc tính/nguy hiểm
    venom = extract_venom_info(comment)
    if venom:
        records.append(make_record(binomial_name, "venom_danger", url, f"Venom status: {venom}"))

    return records, "ok", url

def load_existing_records():
    """Đọc knowledge hiện có để giữ dữ liệu từ VietnamSnakes."""
    if not OUTPUT_JSONL.exists():
        return []

    records = []
    with OUTPUT_JSONL.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    return records

def save_records(existing, new_records, target_species):
    """Ghi knowledge mới và tránh duplicate khi chạy lại collector."""
    target_species = set(target_species)
    existing = [
        record for record in existing
        if not (record.get("source") == SOURCE and record.get("binomial_name") in target_species)
    ]

    all_records = existing + new_records
    with OUTPUT_JSONL.open("w", encoding="utf-8") as f:
        for record in all_records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    return len(all_records)

def save_report(report):
    """Lưu báo cáo kết quả crawl."""
    REPORT_CSV.parent.mkdir(parents=True, exist_ok=True)

    with REPORT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["binomial_name", "status", "sections", "url"])
        writer.writeheader()
        writer.writerows(report)

def main():
    OUTPUT_JSONL.parent.mkdir(parents=True, exist_ok=True)
    REPORT_CSV.parent.mkdir(parents=True, exist_ok=True)

    missing_species = load_missing_species()
    existing = load_existing_records()
    session = create_session()
    new_records, report = [], []

    print(f"Missing species: {len(missing_species)}")

    for i, species in enumerate(missing_species, 1):
        print(f"[{i}/{len(missing_species)}] {species}")

        try:
            records, status, url = collect_species(session, species)
        except requests.RequestException as exc:
            records, status, url = [], f"request_error:{type(exc).__name__}", species_url(species)

        sections = sorted({record["section"] for record in records})
        new_records.extend(records)
        report.append({"binomial_name": species, "status": status, "sections": ",".join(sections), "url": url or ""})

        print(f"  -> {status}: {', '.join(sections) if sections else 'no sections'}")
        time.sleep(REQUEST_DELAY)

    total = save_records(existing, new_records, missing_species)
    save_report(report)
    matched = sum(row["status"] == "ok" for row in report)

    print("\nCOLLECTION COMPLETE")
    print(f"Target species        : {len(missing_species)}")
    print(f"Species found         : {matched}")
    print(f"New knowledge records : {len(new_records)}")
    print(f"Total JSONL records   : {total}")
    print(f"Knowledge JSONL       : {OUTPUT_JSONL}")
    print(f"Report                : {REPORT_CSV}")

if __name__ == "__main__":
    main()