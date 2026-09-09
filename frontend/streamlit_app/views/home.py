import streamlit as st


def render_home_page():
    """Hiển thị trang giới thiệu tổng quan về SnakeGuard AI."""
    st.title("🐍 SnakeGuard AI")
    st.write(
        "SnakeGuard AI là hệ thống hỗ trợ nhận diện, tra cứu và tìm kiếm thông tin về các loài rắn "
        "bằng Computer Vision, Agentic RAG workflow và Multimodal Retrieval."
    )

    st.subheader("🎯 Mục tiêu dự án")
    st.write(
        "Dự án hướng tới việc hỗ trợ người dùng nhận diện rắn nhanh hơn từ hình ảnh hoặc mô tả, "
        "đồng thời cung cấp thông tin tham khảo về đặc điểm, phân bố, độc tính, tập tính và xử trí khi bị rắn cắn."
    )

    st.info("Hệ thống hiện hỗ trợ dữ liệu của 109 loài rắn.")

    st.subheader("⚙️ Các tính năng chính")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.html("""
        <div class="feature-card feature-card-main">
            <h3>🐍 Nhận diện rắn</h3>
            <p>Tải ảnh lên để phát hiện vị trí con rắn và dự đoán loài bằng các mô hình Computer Vision, kèm độ tin cậy phát hiện và phân loại.</p>
        </div>
        """)

    with col2:
        st.html("""
        <div class="feature-card feature-card-main">
            <h3>💬 Chatbot</h3>
            <p>Hỏi đáp kiến thức về loài rắn, độc tính, môi trường sống, tập tính và sơ cứu thông qua Agentic RAG workflow.</p>
            <p><b>⚡ Fast:</b> ưu tiên tốc độ, phù hợp với câu hỏi rõ ràng.</p>
            <p><b>🧠 Deep:</b> truy xuất và đánh giá sâu hơn cho câu hỏi phức tạp hoặc chưa rõ ràng.</p>
        </div>
        """)

    with col3:
        st.html("""
        <div class="feature-card feature-card-main">
            <h3>🔎 Tìm rắn theo mô tả</h3>
            <p>Tìm và xếp hạng các loài phù hợp từ mô tả về màu sắc, hình dáng, môi trường sống hoặc tập tính.</p>
            <p><b>⚡ Fast:</b> kết hợp kết quả truy xuất để phản hồi nhanh.</p>
            <p><b>🧠 Deep:</b> rerank các ứng viên để đưa ra kết quả chính xác hơn.</p>
        </div>
        """)

    col4, col5 = st.columns(2)

    with col4:
        st.html("""
        <div class="feature-card feature-card-secondary">
            <h3>📚 Tra cứu các loài rắn</h3>
            <p>Xem danh sách các loài hiện có trong hệ thống, tên tiếng Việt, tên khoa học, họ, chi và thông tin về nhóm rắn độc có ý nghĩa y khoa.</p>
        </div>
        """)

    with col5:
        st.html("""
        <div class="feature-card feature-card-secondary">
            <h3>🕘 Lịch sử nhận diện</h3>
            <p>Xem lại hình ảnh và kết quả của các lần nhận diện trước, đồng thời lựa chọn và xoá các bản ghi không còn cần thiết.</p>
        </div>
        """)

    st.subheader("📚 Nguồn dữ liệu")
    st.write(
        "Thông tin về các loài rắn được tổng hợp chủ yếu từ VietnamSnakes và The Reptile Database. "
        "Thông tin liên quan đến rắn cắn và sơ cứu ưu tiên nguồn từ Bộ Y tế Việt Nam, "
        "với các nguồn y khoa khác được sử dụng để bổ sung khi cần."
    )

    st.caption(
        "⚠️ SnakeGuard AI chỉ cung cấp thông tin hỗ trợ và không thay thế tư vấn, "
        "chẩn đoán hoặc điều trị y tế chuyên nghiệp."
    )