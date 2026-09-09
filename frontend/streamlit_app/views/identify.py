import requests
import streamlit as st

from components.prediction_result import render_other_predictions, render_species_card
from services.api_client import get_species, predict_snake
from utils.image_utils import draw_detection_box


def render_identify_page():
    """Hiển thị giao diện nhận diện rắn từ ảnh."""
    _initialize_identify_state()
    running = st.session_state.identify_is_running

    st.title("🐍 Identify Snake")
    st.write("Tải ảnh lên để SnakeGuard AI nhận diện loài rắn.")
    st.caption("💡 Để có kết quả tốt nhất, ảnh nên chỉ chứa một con rắn và đối tượng cần nhìn thấy rõ.")

    uploaded_file = st.file_uploader(
        "Chọn ảnh",
        type=["jpg", "jpeg", "png", "webp", "bmp", "tiff"],
        key="identify_upload",
    )

    if uploaded_file:
        st.caption(f"📎 Đã chọn: {uploaded_file.name}")

        if st.button("🔍 Đang nhận diện..." if running else "🔍 Nhận diện", type="primary", width="stretch", disabled=running):
            st.session_state.identify_pending = {
                "image": uploaded_file.getvalue(),
                "name": uploaded_file.name,
                "type": uploaded_file.type,
            }
            st.session_state.identify_result = None
            st.session_state.identify_is_running = True
            st.rerun()

    if running:
        _run_identification()

    _render_identification_result()


def _initialize_identify_state():
    """Khởi tạo trạng thái trang nhận diện."""
    if "identify_is_running" not in st.session_state:
        st.session_state.identify_is_running = False

    if "identify_pending" not in st.session_state:
        st.session_state.identify_pending = None

    if "identify_result" not in st.session_state:
        st.session_state.identify_result = None


def _run_identification():
    """Gọi API nhận diện và lưu kết quả vào session."""
    pending = st.session_state.identify_pending

    if not pending:
        st.session_state.identify_is_running = False
        st.rerun()

    try:
        with st.spinner("Đang phân tích ảnh..."):
            result = predict_snake(pending["image"], pending["name"], pending["type"])

        st.session_state.identify_result = {
            "result": result,
            "image": pending["image"],
        }

    except requests.RequestException as exc:
        st.session_state.identify_result = {"error": f"Không thể gọi API: {exc}"}

    st.session_state.identify_pending = None
    st.session_state.identify_is_running = False
    st.rerun()


def _render_identification_result():
    """Hiển thị kết quả nhận diện đã hoàn tất."""
    state = st.session_state.identify_result

    if not state:
        return

    if state.get("error"):
        st.error(state["error"])
        return

    result = state["result"]

    if not result["detected"]:
        st.warning("Không phát hiện thấy rắn trong ảnh.")
        return

    image_bytes = state["image"]
    species = result["species"]
    detection = result["detection"]
    predictions = result["predictions"]
    detected_image = draw_detection_box(image_bytes, detection["bbox"])

    st.success("✅ Đã phát hiện rắn")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Ảnh tải lên")
        st.image(image_bytes, width="stretch")

    with col2:
        st.markdown("#### Kết quả nhận diện")
        st.image(detected_image, width="stretch")

    if species and predictions:
        render_species_card(species, detection["confidence"], predictions[0]["confidence"])

    if len(predictions) > 1:
        species_list = get_species()
        render_other_predictions(predictions, species_list)