import html

import requests
import streamlit as st
from services.api_client import search_snakes_by_description


def render_search_page():
    """Hiển thị trang tìm rắn theo mô tả."""
    _initialize_search_state()
    searching = st.session_state.search_is_running

    st.title("🔎 Tìm rắn theo mô tả")
    st.caption("Mô tả đặc điểm, môi trường sống hoặc tập tính của con rắn để tìm các loài phù hợp.")

    st.markdown("### Mô tả con rắn")

    description = st.text_area(
        "Mô tả con rắn",
        placeholder="Ví dụ: rắn màu xanh, thân mảnh, sống trên cây...",
        height=120,
        disabled=searching,
        key="search_description",
        label_visibility="collapsed",
    )

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("**Chế độ tìm kiếm**")
        search_mode = st.segmented_control(
            "Chế độ tìm kiếm",
            options=["fast", "deep"],
            format_func=lambda mode: "⚡ Fast" if mode == "fast" else "🧠 Deep",
            default="fast",
            label_visibility="collapsed",
            disabled=searching,
            key="search_mode",
        )

    with col2:
        top_k = st.selectbox(
            "Số kết quả",
            options=[1, 2, 3, 4, 5],
            index=4,
            disabled=searching,
            key="search_top_k",
        )

    if st.button(
        "🔎 Tìm kiếm",
        type="primary",
        width="stretch",
        disabled=searching,
        key="search_submit",
    ):
        if len(description.strip()) < 3:
            st.warning("Hãy nhập mô tả chi tiết hơn về con rắn.")
        else:
            st.session_state.search_pending = {
                "description": description.strip(),
                "search_mode": search_mode or "fast",
                "top_k": top_k,
            }
            st.session_state.search_results = None
            st.session_state.search_error = None
            st.session_state.search_is_running = True
            st.rerun()

    if searching:
        _run_pending_search()

    _render_search_feedback()


def _initialize_search_state():
    """Khởi tạo state của Search by Description."""
    defaults = {
        "search_is_running": False,
        "search_pending": None,
        "search_results": None,
        "search_error": None,
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _run_pending_search():
    """Thực hiện request đang chờ và mở lại controls khi hoàn tất."""
    pending = st.session_state.search_pending

    if not pending:
        st.session_state.search_is_running = False
        return

    try:
        with st.spinner("Đang phân tích mô tả và tìm kiếm loài phù hợp..."):
            response = search_snakes_by_description(
                pending["description"],
                pending["search_mode"],
                pending["top_k"],
            )

        st.session_state.search_results = response.get("results", [])

    except requests.HTTPError as exc:
        st.session_state.search_error = ("warning", _get_http_error(exc))
    except requests.RequestException as exc:
        st.session_state.search_error = ("error", f"Không thể kết nối tới SnakeGuard API: {exc}")

    st.session_state.search_is_running = False
    st.session_state.search_pending = None
    st.rerun()


def _render_search_feedback():
    """Hiển thị lỗi hoặc kết quả tìm kiếm đã lưu."""
    error = st.session_state.search_error
    results = st.session_state.search_results

    if error:
        level, message = error
        st.warning(message) if level == "warning" else st.error(message)
        return

    if results is None:
        return

    if not results:
        st.info("Không tìm thấy loài rắn phù hợp với mô tả.")
        return

    st.subheader(f"Kết quả phù hợp ({len(results)})")
    _render_results(results)


def _render_results(results: list[dict]):
    """Hiển thị danh sách species theo thứ hạng."""
    for index, result in enumerate(results, start=1):
        with st.container(key=f"search_result_{index}"):
            st.markdown(f'<span class="search-rank">Top {index}</span>', unsafe_allow_html=True)

            image_col, info_col = st.columns([1.05, 1.95], vertical_alignment="center")

            with image_col:
                image_url = result.get("image_url")
                if image_url:
                    st.image(image_url, width="stretch")
                else:
                    st.markdown('<div class="search-no-image">Không có ảnh</div>', unsafe_allow_html=True)

            with info_col:
                vietnamese_name = html.escape(result.get("vietnamese_name") or "Chưa có tên tiếng Việt")
                binomial_name = html.escape(result.get("binomial_name") or "N/A")
                venom_text = "Rắn độc nguy hiểm" if result.get("is_mivs") else "Không thuộc nhóm độc nguy hiểm"
                venom_class = "danger" if result.get("is_mivs") else "safe"

                st.markdown(f'<div class="search-title">{vietnamese_name}</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="search-binomial-name">{binomial_name}</div>', unsafe_allow_html=True)
                st.markdown(
                    f'<span class="search-venom-badge {venom_class}">{venom_text}</span>',
                    unsafe_allow_html=True,
                )


def _get_http_error(exc: requests.HTTPError) -> str:
    """Lấy detail từ lỗi Search API."""
    response = exc.response

    if response is not None:
        try:
            detail = response.json().get("detail")
            if detail:
                return detail
        except ValueError:
            pass

    return f"Search API trả về lỗi: {exc}"