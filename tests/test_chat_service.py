import backend.app.services.chat_service as chat_service


def make_result(scope_type="species", section="identification"):
    return {
        "source": "Test Source",
        "source_url": "https://example.com",
        "scope_type": scope_type,
        "scope_value": "Bungarus fasciatus",
        "section": section,
        "content": "Test content",
    }


def make_retrieval(results=None, is_ambiguous=False, candidate_species=None, resolved_species=None, detected_sections=None):
    return {
        "results": results if results is not None else [],
        "is_ambiguous": is_ambiguous,
        "candidate_species": candidate_species or [],
        "resolved_species": resolved_species,
        "detected_sections": detected_sections or [],
    }


def test_resolved_species_uses_knowledge_base(monkeypatch):
    """Câu hỏi về rắn + context đủ -> dùng KB."""
    retrieval = make_retrieval(
        results=[make_result()],
        resolved_species={"binomial_name": "Bungarus fasciatus", "vietnamese_name": "Rắn cạp nong"},
        detected_sections=["identification"],
    )

    monkeypatch.setattr(chat_service, "classify_message", lambda message: "snake")
    monkeypatch.setattr(chat_service, "search_knowledge", lambda *args, **kwargs: retrieval)
    monkeypatch.setattr(chat_service, "generate_answer", lambda *args, **kwargs: {"can_answer": True, "answer": "Rắn cạp nong có các khoang vàng đen rõ rệt."})

    response = chat_service.chat(object(), "Rắn cạp nong có đặc điểm gì?")

    assert response["source_type"] == "knowledge_base"
    assert response["answer"] == "Rắn cạp nong có các khoang vàng đen rõ rệt."
    assert response["is_ambiguous"] is False
    assert response["needs_clarification"] is False
    assert len(response["sources"]) == 1


def test_ambiguous_species_specific_asks_clarification(monkeypatch):
    """Nhiều candidate + species-specific -> hỏi lại user."""
    retrieval = make_retrieval(
        results=[make_result(section="distribution")],
        is_ambiguous=True,
        candidate_species=[
            "Rắn cạp nia Nam, Rắn mai bạc (Bungarus candidus)",
            "Rắn cạp nia bắc (Bungarus multicinctus)",
        ],
        detected_sections=["distribution"],
    )

    monkeypatch.setattr(chat_service, "classify_message", lambda message: "snake")
    monkeypatch.setattr(chat_service, "search_knowledge", lambda *args, **kwargs: retrieval)

    response = chat_service.chat(object(), "Rắn cạp nia phân bố ở đâu?")

    assert response["source_type"] == "knowledge_base"
    assert response["is_ambiguous"] is True
    assert response["needs_clarification"] is True
    assert "Rắn cạp nia Nam" in response["answer"]
    assert "Rắn cạp nia bắc" in response["answer"]
    assert response["answer"].endswith("được không?")


def test_ambiguous_medical_with_shared_scope_can_answer(monkeypatch):
    """Ambiguous medical + scope genus -> vẫn có thể trả lời từ KB."""
    retrieval = make_retrieval(
        results=[make_result(scope_type="genus", section="diagnosis")],
        is_ambiguous=True,
        candidate_species=[
            "Rắn cạp nia Nam (Bungarus candidus)",
            "Rắn cạp nia bắc (Bungarus multicinctus)",
        ],
        detected_sections=["diagnosis"],
    )

    monkeypatch.setattr(chat_service, "classify_message", lambda message: "snake")
    monkeypatch.setattr(chat_service, "search_knowledge", lambda *args, **kwargs: retrieval)
    monkeypatch.setattr(chat_service, "generate_answer", lambda *args, **kwargs: {"can_answer": True, "answer": "Nhóm Bungarus có thể gây liệt thần kinh."})

    response = chat_service.chat(object(), "Rắn cạp nia cắn có triệu chứng gì?")

    assert response["source_type"] == "knowledge_base"
    assert response["is_ambiguous"] is True
    assert response["needs_clarification"] is False
    assert response["answer"] == "Nhóm Bungarus có thể gây liệt thần kinh."


def test_empty_results_fallbacks_to_web(monkeypatch):
    """Không retrieve được context -> web."""
    retrieval = make_retrieval()

    monkeypatch.setattr(chat_service, "classify_message", lambda message: "snake")
    monkeypatch.setattr(chat_service, "search_knowledge", lambda *args, **kwargs: retrieval)
    monkeypatch.setattr(chat_service, "search_web", lambda message: {"answer": "Thông tin từ Internet.", "sources": [{"source": "Web Source", "source_url": "https://example.com"}]})

    response = chat_service.chat(object(), "Một loài rắn chưa có trong KB")

    assert response["source_type"] == "web"
    assert response["needs_clarification"] is False
    assert response["notice"] == "Tôi đã tìm thêm thông tin trên Internet."
    assert "Tôi đã kiểm tra cơ sở kiến thức hiện có" in response["answer"]
    assert "Thông tin từ Internet." in response["answer"]


def test_insufficient_context_fallbacks_to_web(monkeypatch):
    """Có chunks nhưng LLM đánh giá không đủ -> web."""
    retrieval = make_retrieval(
        results=[make_result(section="identification")],
        detected_sections=["identification"],
    )

    monkeypatch.setattr(chat_service, "classify_message", lambda message: "snake")
    monkeypatch.setattr(chat_service, "search_knowledge", lambda *args, **kwargs: retrieval)
    monkeypatch.setattr(chat_service, "generate_answer", lambda *args, **kwargs: {"can_answer": False, "answer": None})
    monkeypatch.setattr(chat_service, "search_web", lambda message: {"answer": "Rắn lục đầu bạc có tên khoa học là Azemiops feae.", "sources": []})

    response = chat_service.chat(object(), "Rắn lục đầu bạc có đặc điểm gì?")

    assert response["source_type"] == "web"
    assert response["notice"] == "Tôi đã tìm thêm thông tin trên Internet."
    assert "Azemiops feae" in response["answer"]


def test_casual_message_skips_rag_and_web(monkeypatch):
    """Casual -> Groq trực tiếp, không retrieval và không web."""
    monkeypatch.setattr(chat_service, "classify_message", lambda message: "casual")
    monkeypatch.setattr(chat_service, "generate_casual_answer", lambda message: "Ừ, nghe cũng dễ chịu đấy 😄")

    def should_not_run(*args, **kwargs):
        raise AssertionError("Không được gọi hàm này với casual message.")

    monkeypatch.setattr(chat_service, "search_knowledge", should_not_run)
    monkeypatch.setattr(chat_service, "search_web", should_not_run)

    response = chat_service.chat(object(), "Trời hôm nay đẹp nhỉ?")

    assert response["source_type"] == "casual"
    assert response["answer"] == "Ừ, nghe cũng dễ chịu đấy 😄"
    assert response["sources"] == []
    assert response["is_ambiguous"] is False
    assert response["needs_clarification"] is False
    assert response["notice"] is None


def test_web_unavailable_does_not_crash(monkeypatch):
    """KB không đủ + web lỗi -> graceful fallback."""
    retrieval = make_retrieval(results=[make_result()], detected_sections=["identification"])

    monkeypatch.setattr(chat_service, "classify_message", lambda message: "snake")
    monkeypatch.setattr(chat_service, "search_knowledge", lambda *args, **kwargs: retrieval)
    monkeypatch.setattr(chat_service, "generate_answer", lambda *args, **kwargs: {"can_answer": False, "answer": None})

    def raise_web_error(*args, **kwargs):
        raise chat_service.WebSearchUnavailableError("Web unavailable")

    monkeypatch.setattr(chat_service, "search_web", raise_web_error)

    response = chat_service.chat(object(), "Rắn lục đầu bạc sống ở đâu?")

    assert response["source_type"] == "unavailable"
    assert response["sources"] == []
    assert response["needs_clarification"] is False
    assert response["notice"] == "Tìm kiếm Internet hiện không khả dụng."