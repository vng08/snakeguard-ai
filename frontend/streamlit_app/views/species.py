import pandas as pd
import requests
import streamlit as st

from services.api_client import get_species


def render_species_page():
    """Hiển thị danh sách toàn bộ loài rắn."""
    st.title("📚 Danh sách Các loài rắn")
    st.caption("Danh sách các loài rắn hiện có trong hệ thống.")

    try:
        with st.spinner("Đang tải danh sách loài rắn..."):
            species_list = get_species()

    except requests.RequestException as exc:
        st.error(f"Không thể kết nối tới SnakeGuard API: {exc}")
        return

    if not species_list:
        st.info("Hiện chưa có dữ liệu loài rắn.")
        return

    total_species = len(species_list)
    total_medically_important = sum(
        1 for species in species_list if species.get("is_mivs")
    )
    total_other = total_species - total_medically_important

    with st.container(key="species_stats"):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Tổng số loài", total_species)

        with col2:
            st.metric(
                "Rắn độc có ý nghĩa y khoa",
                total_medically_important,
            )

        with col3:
            st.metric(
                "Không thuộc nhóm độc nguy hiểm",
                total_other,
            )

    st.markdown("### Tìm loài rắn")

    keyword = st.text_input(
        "Tìm loài rắn",
        placeholder="Nhập tên tiếng Việt, tên khoa học, họ hoặc chi...",
        key="species_search",
        label_visibility="collapsed",
    ).strip().lower()

    filtered_species = _filter_species(species_list, keyword)
    dataframe = _build_species_dataframe(filtered_species)

    st.markdown(f"**Kết quả hiển thị: {len(filtered_species)} loài**")

    with st.container(key="species_table"):
        st.markdown(
            dataframe.to_html(
                index=False,
                escape=True,
                classes="species-data-table",
            ),
            unsafe_allow_html=True,
        )


def _filter_species(
    species_list: list[dict],
    keyword: str,
) -> list[dict]:
    """Lọc danh sách loài theo từ khoá."""
    if not keyword:
        return species_list

    results = []

    for species in species_list:
        searchable_text = " ".join(
            [
                str(species.get("binomial_name", "")),
                str(species.get("vietnamese_name", "")),
                str(species.get("family", "")),
                str(species.get("genus", "")),
            ]
        ).lower()

        if keyword in searchable_text:
            results.append(species)

    return results


def _build_species_dataframe(
    species_list: list[dict],
) -> pd.DataFrame:
    """Chuyển dữ liệu species sang bảng tiếng Việt."""
    rows = []

    for species in species_list:
        rows.append(
            {
                "ID": species.get("id"),
                "Tên khoa học": species.get("binomial_name") or "Chưa có",
                "Tên tiếng Việt": species.get("vietnamese_name") or "Chưa có",
                "Họ": species.get("family") or "Chưa có",
                "Chi": species.get("genus") or "Chưa có",
                "Nhóm độc": (
                    "Rắn độc có ý nghĩa y khoa"
                    if species.get("is_mivs")
                    else "Không thuộc nhóm độc nguy hiểm"
                ),
            }
        )

    return pd.DataFrame(rows)