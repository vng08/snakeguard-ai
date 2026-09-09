import re
import time
import uuid

import requests
import streamlit as st

from services.api_client import delete_chat_history, get_chat_history, send_chat_message


def render_assistant_page():
    """Hiển thị trang Chatbot."""
    _initialize_chat()
    running = st.session_state.chat_is_running

    col1, col2 = st.columns([4, 1])

    with col1:
        st.title("💬 Chatbot")
        st.caption("Hỏi đáp kiến thức về rắn, nhận dạng loài, sơ cứu và an toàn.")

    with col2:
        st.write("")
        if st.button("＋ Cuộc trò chuyện mới", width="stretch", disabled=running):
            _confirm_new_conversation()

    st.markdown("**Chế độ tìm kiếm**")

    search_mode = st.segmented_control(
        "Chế độ tìm kiếm",
        options=["fast", "deep"],
        format_func=lambda mode: "⚡ Fast" if mode == "fast" else "🧠 Deep",
        default="fast",
        label_visibility="collapsed",
        disabled=running,
        key="chat_search_mode",
    )

    _render_messages()

    prompt = st.chat_input("Hỏi Chatbot...", disabled=running)

    if prompt and not running:
        _queue_message(prompt, search_mode or "fast")

    if running:
        _run_pending_message()


def _initialize_chat():
    """Khởi tạo session, lịch sử và trạng thái chat."""
    if "chat_session_id" not in st.session_state:
        st.session_state.chat_session_id = str(uuid.uuid4())

    if "chat_messages" not in st.session_state:
        try:
            history = get_chat_history(st.session_state.chat_session_id)
            st.session_state.chat_messages = [
                {"role": message["role"], "content": message["content"], "sources": []}
                for message in history
            ]
        except requests.RequestException:
            st.session_state.chat_messages = []

    if "chat_is_running" not in st.session_state:
        st.session_state.chat_is_running = False

    if "chat_pending" not in st.session_state:
        st.session_state.chat_pending = None


def _queue_message(prompt: str, search_mode: str):
    """Đưa message vào hàng chờ và khoá chat trước khi gọi backend."""
    st.session_state.chat_messages.append({
        "role": "user",
        "content": prompt,
        "sources": [],
    })

    st.session_state.chat_pending = {
        "message": prompt,
        "search_mode": search_mode,
    }
    st.session_state.chat_is_running = True
    st.rerun()


def _run_pending_message():
    """Gửi message đang chờ và xử lý phản hồi từ backend."""
    pending = st.session_state.chat_pending

    if not pending:
        st.session_state.chat_is_running = False
        st.rerun()

    with st.chat_message("assistant"):
        try:
            with st.spinner("Đang tìm kiếm và phân tích..."):
                response = send_chat_message(
                    st.session_state.chat_session_id,
                    pending["message"],
                    pending["search_mode"],
                )

            answer = response["answer"]
            sources = response.get("sources", [])

            _stream_assistant_answer(answer)
            _render_sources(sources)

            st.session_state.chat_messages.append({
                "role": "assistant",
                "content": answer,
                "sources": sources,
            })

        except (requests.RequestException, KeyError, TypeError, ValueError):
            error_message = (
                "Hệ thống hiện đang gặp sự cố hoặc chưa thể xử lý yêu cầu này. "
                "Vui lòng thử lại sau."
            )

            st.markdown(error_message)

            st.session_state.chat_messages.append({
                "role": "assistant",
                "content": error_message,
                "sources": [],
            })

    st.session_state.chat_pending = None
    st.session_state.chat_is_running = False
    st.rerun()


def _render_messages():
    """Hiển thị toàn bộ lịch sử hội thoại."""
    for message in st.session_state.chat_messages:
        role = message["role"]

        with st.chat_message(role):
            st.markdown(message["content"])

            if role == "assistant":
                _render_sources(message.get("sources", []))


def _stream_assistant_answer(answer: str):
    """Hiển thị câu trả lời theo hiệu ứng streaming và giữ Markdown."""
    placeholder = st.empty()
    current_text = ""

    for chunk in re.findall(r"\S+\s*", answer):
        current_text += chunk
        placeholder.markdown(current_text)
        time.sleep(0.015)


def _render_sources(sources: list[dict]):
    """Hiển thị nguồn được sử dụng trong câu trả lời."""
    if not sources:
        return

    with st.expander(f"Nguồn tham khảo ({len(sources)})"):
        for source in sources:
            name = source.get("source", "Nguồn tham khảo")
            url = source.get("source_url")

            if url:
                st.markdown(f"- [{name}]({url})")
            else:
                st.markdown(f"- {name}")


@st.dialog("Tạo cuộc trò chuyện mới")
def _confirm_new_conversation():
    """Xác nhận trước khi xoá cuộc trò chuyện hiện tại."""
    st.warning(
        "Toàn bộ lịch sử và dữ liệu của cuộc trò chuyện hiện tại sẽ bị xoá "
        "và không thể khôi phục."
    )
    st.write("Bạn có chắc chắn muốn tiếp tục không?")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("Huỷ", width="stretch"):
            st.rerun()

    with col2:
        if st.button("Tạo cuộc trò chuyện mới", type="primary", width="stretch"):
            _new_conversation()


def _new_conversation():
    """Xoá conversation hiện tại và tạo session mới."""
    try:
        delete_chat_history()
        st.session_state.chat_session_id = str(uuid.uuid4())
        st.session_state.chat_messages = []
        st.session_state.chat_pending = None
        st.session_state.chat_is_running = False
        st.rerun()
    except requests.RequestException as exc:
        st.error(f"Không thể tạo cuộc trò chuyện mới: {exc}")