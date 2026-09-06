import json
import re
import time
from pathlib import Path
from urllib.parse import urljoin

import pandas as pd
import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

ROOT_DIR = Path(__file__).resolve().parents[1]

# Đường dẫn input/output
SPECIES_CSV = ROOT_DIR / "data/processed/species_metadata.csv"
OUTPUT_JSONL = ROOT_DIR / "data/raw/species_knowledge.jsonl"
MISSING_CSV = ROOT_DIR / "data/reports/vietnamsnakes_missing.csv"

# Cấu hình VietnamSnakes
BASE_URL = "https://vietnamsnakes.com"
ALL_SPECIES_URL = f"{BASE_URL}/all-of-species"
REQUEST_DELAY = 1

# Một số tên trên website khác với class canonical của model
ALIAS_MAP = {
    "Bungarus fasciatus": "Bungarus bifasciatus",
    "Bungarus multicinctus": "Bungarus wanghaotingi",
    "Cylindrophis ruffus": "Cylindrophis jodiae",
    "Homalopsis buccata": "Homalopsis mereljcoxi",
    "Sinomicrurus macclellandi": "Sinomicrurus macclellandii",
    "Trimeresurus guoi": "Trimeresurus albolabris guoi",
}

# Heading thật trong tab #nav-detail
SECTION_HEADINGS = {
    "đặc điểm nhận dạng": "identification",
    "thông tin độc tố": "venom_detail",
    "tập tính hành vi": "habitat_behavior",
    "phân bố": "distribution",
}


def clean_text(text: str) -> str:
    """Chuẩn hóa khoảng trắng."""
    return re.sub(r"\s+", " ", text).strip()


def remove_exact_repeat(text: str) -> str:
    """Loại trường hợp website lặp nguyên một đoạn hai lần."""
    text = clean_text(text)
    middle = len(text) // 2

    for split in range(max(1, middle - 3), min(len(text), middle + 4)):
        left, right = text[:split].strip(), text[split:].strip()
        if left and left.casefold() == right.casefold():
            return left

    return text


def create_session():
    """Tạo HTTP session có retry khi website phản hồi lỗi."""
    session = requests.Session()
    retry = Retry(total=3, connect=3, read=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504], allowed_methods=["GET"])
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session.headers.update({"User-Agent": "SnakeGuardAI/1.0 (educational AI project; knowledge collection)"})
    return session


def normalize_name(name: str) -> str:
    """Chuẩn hóa tên khoa học để so khớp dataset với website."""
    name = re.sub(r"\bcf\.\s*", "", name)
    return clean_text(name)


def normalize_heading(text: str) -> str:
    """Chuẩn hóa heading/label trước khi so khớp."""
    return clean_text(text).lower().rstrip(":：")


def detect_section(text: str):
    """Map heading trên VietnamSnakes về section chuẩn."""
    normalized = normalize_heading(text)

    for heading, section in SECTION_HEADINGS.items():
        if normalized.startswith(heading):
            return section

    return None


def extract_profile_field(soup, label):
    """Lấy đúng badge của Nọc độc hoặc Mức độ đe dọa."""
    target = normalize_heading(label)
    information = soup.select_one("#species-page .information")

    if not information:
        return None

    for row in information.select("div.d-flex.align-items-center.gap-1"):
        label_node = row.select_one(".fw-600")
        badge_node = row.select_one(".badge")
        if not label_node or not badge_node:
            continue

        if normalize_heading(label_node.get_text(" ", strip=True)) == target:
            return clean_text(badge_node.get_text(" ", strip=True))

    return None


def extract_distribution_fallback(soup):
    """Lấy Phân bố/Vị trí ở phần thông tin chung nếu #nav-detail không có."""
    information = soup.select_one("#species-page .information")
    if not information:
        return None

    for label_node in information.select(".fw-600"):
        label = normalize_heading(label_node.get_text(" ", strip=True))
        if label not in {"phân bố", "vị trí"}:
            continue

        parent = label_node.parent
        if not parent:
            continue

        values = [clean_text(text) for text in parent.stripped_strings]
        values = [text for text in values if normalize_heading(text) != label]

        if values:
            return clean_text(" ".join(values))

    return None


def extract_detail_sections(soup):
    """Lấy các section thật trong tab Chi tiết (#nav-detail)."""
    container = soup.select_one("#nav-detail")
    if not container:
        return {}

    sections = {}
    blocks = container.select("div.w-100.text-white > div.mb-4")

    # Fallback nếu wrapper thay đổi nhưng cấu trúc section vẫn là mb-4
    if not blocks:
        blocks = container.select("div.mb-4")

    for block in blocks:
        heading_node = block.select_one("div.mb-2.h5")
        if not heading_node:
            continue

        section = detect_section(heading_node.get_text(" ", strip=True))
        if not section:
            continue

        content_node = heading_node.find_next_sibling("div")
        if not content_node:
            continue

        content = remove_exact_repeat(content_node.get_text(" ", strip=True))
        if content:
            sections[section] = content

    return sections


def build_vietnam_snakes_mapping(session):
    """Map tên khoa học trên VietnamSnakes về URL species."""
    response = session.get(ALL_SPECIES_URL, timeout=(10, 45))
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")
    mapping = {}

    for element in soup.find_all("a"):
        binomial_tag, href = element.find("i"), element.get("href")
        if binomial_tag is None or not href:
            continue

        binomial_name = binomial_tag.get_text(" ", strip=True)
        if len(binomial_name.split()) < 2:
            continue

        normalized = normalize_name(binomial_name)
        if normalized not in mapping:
            mapping[normalized] = urljoin(BASE_URL, href)

    return mapping


def find_species_match(binomial_name, vn_mapping):
    """Ưu tiên exact match, nếu không có thì thử alias."""
    url = vn_mapping.get(normalize_name(binomial_name))
    if url:
        return url, "exact"

    alias = ALIAS_MAP.get(binomial_name)
    if alias:
        url = vn_mapping.get(normalize_name(alias))
        if url:
            return url, "alias"

    return None, None


def extract_species_sections(session, url):
    """Lấy toàn bộ knowledge có cấu trúc từ trang species."""
    response = session.get(url, timeout=(10, 45))
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")

    # Identification, Venomous Detail, Behavior, Distribution
    sections = extract_detail_sections(soup)

    # Một số trang chỉ có Phân bố/Vị trí ở phần thông tin chung
    if "distribution" not in sections:
        distribution = extract_distribution_fallback(soup)
        if distribution:
            sections["distribution"] = distribution

    # Hai field structured ở phần profile
    venom_status = extract_profile_field(soup, "Nọc độc")
    threat_level = extract_profile_field(soup, "Mức độ đe dọa")
    venom_detail = sections.pop("venom_detail", None)

    # Gom thông tin độc tính/nguy hiểm thành một section
    danger_parts = []
    if venom_status:
        danger_parts.append(f"Nọc độc: {venom_status}.")
    if threat_level:
        danger_parts.append(f"Mức độ đe dọa: {threat_level}.")
    if venom_detail:
        danger_parts.append(f"Thông tin độc tố: {venom_detail}")

    if danger_parts:
        sections["venom_danger"] = " ".join(danger_parts)

    return sections


def main():
    OUTPUT_JSONL.parent.mkdir(parents=True, exist_ok=True)
    MISSING_CSV.parent.mkdir(parents=True, exist_ok=True)

    species_df = pd.read_csv(SPECIES_CSV)
    session = create_session()

    print(f"Loaded species: {len(species_df)}")
    print("Building VietnamSnakes URL mapping...")
    vn_mapping = build_vietnam_snakes_mapping(session)
    print(f"VietnamSnakes species discovered: {len(vn_mapping)}")

    records, missing = [], []
    matched_species = exact_matches = alias_matches = 0

    # Crawl toàn bộ class canonical của model
    for index, row in species_df.iterrows():
        binomial_name = row["binomial_name"]
        print(f"[{index + 1}/{len(species_df)}] {binomial_name}")

        url, match_type = find_species_match(binomial_name, vn_mapping)
        if not url:
            print("  -> NOT FOUND")
            missing.append({"binomial_name": binomial_name, "reason": "species_url_not_found"})
            continue

        try:
            sections = extract_species_sections(session, url)
        except requests.RequestException as exc:
            print(f"  -> REQUEST ERROR: {exc}")
            missing.append({"binomial_name": binomial_name, "reason": f"request_error: {type(exc).__name__}"})
            continue

        if not sections:
            print("  -> PAGE FOUND BUT NO SECTIONS")
            missing.append({"binomial_name": binomial_name, "reason": "no_supported_sections"})
            continue

        matched_species += 1
        exact_matches += match_type == "exact"
        alias_matches += match_type == "alias"

        for section, content in sections.items():
            records.append({
                "binomial_name": binomial_name,
                "section": section,
                "source": "VietnamSnakes",
                "source_url": url,
                "content": content,
            })

        print(f"  -> OK ({match_type}): {len(sections)} sections {list(sections)}")
        time.sleep(REQUEST_DELAY)

    # Lưu raw knowledge
    with OUTPUT_JSONL.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    # Lưu report các species chưa crawl được
    pd.DataFrame(missing).to_csv(MISSING_CSV, index=False)

    print("\nCOLLECTION COMPLETE")
    print(f"Total model species : {len(species_df)}")
    print(f"Species matched     : {matched_species}")
    print(f"Exact matches       : {exact_matches}")
    print(f"Alias matches       : {alias_matches}")
    print(f"Species missing     : {len(missing)}")
    print(f"Knowledge records   : {len(records)}")
    print(f"JSONL               : {OUTPUT_JSONL}")
    print(f"Missing report      : {MISSING_CSV}")


if __name__ == "__main__":
    main()