import atexit
from pathlib import Path

import streamlit as st
from services.api_client import delete_chat_history
from views.assistant import render_assistant_page
from views.home import render_home_page
from views.identify import render_identify_page
from views.prediction_logs import render_prediction_logs_page
from views.search import render_search_page
from views.species import render_species_page


def load_css():
    """Load CSS dùng chung và CSS riêng của từng page."""
    styles_dir = Path(__file__).parent / "styles"
    css_files = ["main.css", "home.css", "identify.css", "search.css", "assistant.css", "species.css", "prediction_logs.css"]
    css = "\n".join((styles_dir / filename).read_text() for filename in css_files)
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


def cleanup_chat():
    """Xoá conversation hiện tại khi Streamlit app dừng."""
    try:
        delete_chat_history()
    except Exception:
        pass


st.set_page_config(page_title="SnakeGuard AI", page_icon="🐍", layout="wide")
load_css()

if "_chat_cleanup_registered" not in st.session_state:
    atexit.register(cleanup_chat)
    st.session_state._chat_cleanup_registered = True

pages = [
    st.Page(render_home_page, title="Trang chủ", icon="🏠", default=True),
    st.Page(render_identify_page, title="Nhận diện rắn", icon="🐍"),
    st.Page(render_assistant_page, title="Chatbot", icon="💬"),
    st.Page(render_search_page, title="Tìm rắn theo mô tả", icon="🔎"),
    st.Page(render_species_page, title="Các loài rắn", icon="📚"),
    st.Page(render_prediction_logs_page, title="Lịch sử nhận diện", icon="🕘"),
]

navigation = st.navigation(pages, position="top")
navigation.run()