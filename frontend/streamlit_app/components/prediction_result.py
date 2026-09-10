import html

import streamlit as st


def get_confidence_style(confidence: float) -> tuple[str, str]:
    """Trả về màu và mức đánh giá theo confidence."""
    if confidence >= 0.8:
        return "#16a34a", "Cao"
    if confidence >= 0.5:
        return "#f59e0b", "Trung bình"
    return "#dc2626", "Thấp"


def build_donut(confidence: float, label: str) -> str:
    """Tạo donut biểu diễn độ tin cậy."""
    color, level = get_confidence_style(confidence)
    percent = confidence * 100

    return f"""
    <div class="score-item">
        <div class="score-donut" style="background:conic-gradient({color} {percent}%, #edf2f7 0);">
            <div class="score-donut-inner">
                <strong>{percent:.1f}%</strong>
                <span>{level}</span>
            </div>
        </div>
        <div class="score-label">{label}</div>
    </div>
    """


def render_species_card(species: dict, detection_confidence: float, classification_confidence: float):
    """Hiển thị card loài rắn và độ tin cậy của model."""
    name = html.escape(species["vietnamese_name"] or species["binomial_name"])
    binomial = html.escape(species["binomial_name"])
    family = html.escape(species["family"] or "N/A")
    genus = html.escape(species["genus"] or "N/A")

    venom_text = "Rắn độc nguy hiểm" if species["is_mivs"] else "Không thuộc nhóm độc nguy hiểm"
    venom_class = "danger" if species["is_mivs"] else "safe"

    detection = build_donut(detection_confidence, "Phát hiện")
    classification = build_donut(classification_confidence, "Phân loại")

    st.html(f"""
    <div class="species-card">
        <div class="species-main">
            <div class="species-info">
                <h2>{name}</h2>

                <div class="species-danger">
                    <span class="venom-badge {venom_class}">{venom_text}</span>
                </div>

                <div class="species-meta">
                    <div>
                        <span>Tên khoa học</span>
                        <strong>{binomial}</strong>
                    </div>
                    <div>
                        <span>Họ</span>
                        <strong>{family}</strong>
                    </div>
                    <div>
                        <span>Chi</span>
                        <strong>{genus}</strong>
                    </div>
                </div>
            </div>

            <div class="score-panel">
                <div class="score-panel-title">Độ tin cậy</div>

                <div class="score-grid">
                    {detection}
                    {classification}
                </div>
            </div>
        </div>
    </div>
    """)


def render_other_predictions(predictions: list[dict], species_list: list[dict]):
    """Hiển thị các kết quả dự đoán khác ngoài Top-1."""
    name_map = {item["binomial_name"]: item["vietnamese_name"] for item in species_list}
    st.subheader("Các kết quả dự đoán khác")

    for index, prediction in enumerate(predictions[1:], start=2):
        confidence = prediction["confidence"]
        color, _ = get_confidence_style(confidence)
        binomial = html.escape(prediction["binomial_name"])
        vietnamese_name = html.escape(name_map.get(prediction["binomial_name"]) or "Chưa có tên tiếng Việt")

        st.html(f"""
        <div class="prediction-card">
            <div class="prediction-rank">{index}</div>
            <div class="prediction-info">
                <strong>{vietnamese_name}</strong>
                <span><i>{binomial}</i></span>
            </div>
            <div class="prediction-score" style="color:{color};">{confidence:.2%}</div>
        </div>
        """)