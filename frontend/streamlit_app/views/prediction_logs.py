import base64
import mimetypes
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests
import streamlit as st
from config import VN_TIMEZONE
from services.api_client import (
    delete_all_prediction_logs,
    delete_prediction_log,
    get_prediction_logs,
    get_species,
)


def render_prediction_logs_page():
    """Hiển thị và quản lý lịch sử nhận diện."""
    st.title("🕘 Lịch sử nhận diện")
    st.caption("Xem và quản lý các lần nhận diện rắn đã được lưu trong hệ thống.")

    try:
        with st.spinner("Đang tải lịch sử nhận diện..."):
            logs = get_prediction_logs()
            species_list = get_species()
    except requests.RequestException as exc:
        st.error(f"Không thể tải lịch sử nhận diện: {exc}")
        return

    if isinstance(logs, dict):
        logs = logs.get("logs", [])

    if not logs:
        st.info("Hiện chưa có lịch sử nhận diện.")
        return

    species_by_id = _build_species_map(species_list)

    with st.container(key="prediction_logs_stats"):
        st.metric("Tổng số lần nhận diện", len(logs))

    st.markdown("### Danh sách lịch sử")

    dataframe = _build_logs_dataframe(logs, species_by_id)

    # Chiều cao ôm theo số dòng, nhiều log thì mới cuộn dọc.
    row_height = 72
    table_height = min(540, 40 + len(dataframe) * row_height)

    edited_df = st.data_editor(
        dataframe,
        width="stretch",
        height=table_height,
        row_height=row_height,
        hide_index=True,
        key="prediction_logs_table",
        disabled=["Ảnh", "Loài dự đoán", "Phát hiện", "Phân loại", "Thời gian"],
        column_config={
            "_log_id": None,
            "Chọn": st.column_config.CheckboxColumn("Chọn", width="small"),
            "Ảnh": st.column_config.ImageColumn("Ảnh", width="small"),
            "Loài dự đoán": st.column_config.TextColumn("Loài dự đoán", width="large"),
            "Phát hiện": st.column_config.TextColumn("Phát hiện", width="small"),
            "Phân loại": st.column_config.TextColumn("Phân loại", width="small"),
            "Thời gian": st.column_config.TextColumn("Thời gian", width="medium"),
        },
    )

    selected_ids = edited_df.loc[edited_df["Chọn"], "_log_id"].tolist()
    _render_actions(selected_ids, len(logs))


def _build_logs_dataframe(logs: list[dict], species_by_id: dict) -> pd.DataFrame:
    """Chuyển prediction logs thành dữ liệu hiển thị."""
    rows = []

    for log in logs:
        species_id = log.get("predicted_species_id")

        rows.append({
            "_log_id": log.get("id"),
            "Chọn": False,
            "Ảnh": _prepare_image(log.get("image_url")),
            "Loài dự đoán": species_by_id.get(species_id, "Không xác định") if species_id else "Không xác định",
            "Phát hiện": _format_confidence(log.get("detection_confidence")),
            "Phân loại": _format_confidence(log.get("confidence")),
            "Thời gian": _format_datetime(log.get("created_at")),
        })

    return pd.DataFrame(rows)


def _build_species_map(species_list: list[dict]) -> dict:
    """Tạo map species ID sang tên tiếng Việt."""
    return {
        species["id"]: species.get("vietnamese_name") or species.get("binomial_name") or "Không xác định"
        for species in species_list if species.get("id") is not None
    }


def _format_confidence(value) -> str:
    """Chuyển confidence sang phần trăm."""
    if value is None:
        return "Không có"

    try:
        return f"{float(value) * 100:.2f}%"
    except (TypeError, ValueError):
        return str(value)


def _format_datetime(value) -> str:
    """Chuyển thời gian UTC sang giờ Việt Nam."""
    if not value:
        return "Không có"

    try:
        dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))

        # Backend đang lưu datetime.utcnow nên datetime không timezone được hiểu là UTC.
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)

        return dt.astimezone(VN_TIMEZONE).strftime("%d/%m/%Y %H:%M")
    except ValueError:
        return str(value)


def _prepare_image(image_path: str | None) -> str | None:
    """Chuyển ảnh local thành data URL để hiển thị trong bảng."""
    if not image_path:
        return None

    if image_path.startswith(("http://", "https://", "data:")):
        return image_path

    path = Path(image_path)
    if not path.exists():
        return None

    mime_type = mimetypes.guess_type(path.name)[0] or "image/jpeg"
    encoded = base64.b64encode(path.read_bytes()).decode("utf-8")
    return f"data:{mime_type};base64,{encoded}"


def _render_actions(selected_ids: list[int], total_logs: int):
    """Hiển thị các chức năng xoá lịch sử."""
    col1, col2 = st.columns(2)

    with col1:
        if st.button(f"🗑️ Xoá mục đã chọn ({len(selected_ids)})", type="primary", width="stretch", disabled=not selected_ids):
            _confirm_delete_selected(selected_ids)

    with col2:
        if st.button(f"Xoá toàn bộ ({total_logs})", width="stretch"):
            _confirm_delete_all_logs()


@st.dialog("Xoá lịch sử đã chọn")
def _confirm_delete_selected(log_ids: list[int]):
    """Xác nhận xoá các prediction log đã chọn."""
    st.warning(f"{len(log_ids)} mục đã chọn sẽ bị xoá và không thể khôi phục.")
    st.write("Bạn có chắc chắn muốn tiếp tục không?")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("Huỷ", width="stretch"):
            st.rerun()

    with col2:
        if st.button("Xoá", type="primary", width="stretch"):
            try:
                for log_id in log_ids:
                    delete_prediction_log(int(log_id))
                st.rerun()
            except requests.RequestException as exc:
                st.error(f"Không thể xoá lịch sử: {exc}")


@st.dialog("Xoá toàn bộ lịch sử")
def _confirm_delete_all_logs():
    """Xác nhận xoá toàn bộ prediction logs."""
    st.warning("Toàn bộ lịch sử nhận diện sẽ bị xoá và không thể khôi phục.")
    st.write("Bạn có chắc chắn muốn tiếp tục không?")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("Huỷ", width="stretch"):
            st.rerun()

    with col2:
        if st.button("Xoá toàn bộ", type="primary", width="stretch"):
            try:
                delete_all_prediction_logs()
                st.rerun()
            except requests.RequestException as exc:
                st.error(f"Không thể xoá lịch sử: {exc}")