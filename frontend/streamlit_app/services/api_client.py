import requests
from config import API_BASE_URL, API_TIMEOUT


# Lấy danh sách loài để ánh xạ tên khoa học sang tên tiếng Việt
def get_species() -> list[dict]:
    """Lấy danh sách species và đo thời gian phản hồi API."""
    response = requests.get(f"{API_BASE_URL}/species", timeout=API_TIMEOUT,)
    response.raise_for_status()
    data = response.json()

    return data


# Gửi ảnh sang FastAPI để nhận kết quả nhận diện
def predict_snake(image_bytes: bytes, filename: str, content_type: str) -> dict:
    files = {"file": (filename, image_bytes, content_type)}
    response = requests.post(f"{API_BASE_URL}/predict", files=files, timeout=API_TIMEOUT)
    response.raise_for_status()
    return response.json()


# Gửi message vào Agentic RAG chatbot
def send_chat_message(session_id: str, message: str, search_mode: str = "fast") -> dict:
    payload = {
        "session_id": session_id,
        "message": message,
        "search_mode": search_mode,
    }

    response = requests.post(f"{API_BASE_URL}/chat", json=payload, timeout=API_TIMEOUT)
    response.raise_for_status()
    return response.json()


# Lấy lịch sử conversation hiện tại
def get_chat_history(session_id: str) -> list[dict]:
    response = requests.get(f"{API_BASE_URL}/chat/{session_id}", timeout=API_TIMEOUT)
    response.raise_for_status()
    return response.json()["messages"]


# Tạo conversation mới bằng cách xoá lịch sử hiện tại
def delete_chat_history() -> int:
    response = requests.delete(f"{API_BASE_URL}/chat", timeout=API_TIMEOUT)
    response.raise_for_status()
    return response.json()["deleted_count"]

# Tìm loài rắn dựa trên mô tả
def search_snakes_by_description(description: str, search_mode: str = "fast", top_k: int = 5) -> dict:
    payload = {"description": description, "search_mode": search_mode, "top_k": top_k}
    response = requests.post(f"{API_BASE_URL}/search-by-description", json=payload, timeout=API_TIMEOUT)
    response.raise_for_status()
    return response.json()

def get_prediction_logs() -> list[dict]:
    """Lấy toàn bộ lịch sử nhận diện."""
    response = requests.get(f"{API_BASE_URL}/prediction-logs", timeout=API_TIMEOUT)
    response.raise_for_status()
    return response.json()


# prediction log
def delete_prediction_log(log_id: int) -> dict:
    """Xoá một prediction log theo ID."""
    response = requests.delete(f"{API_BASE_URL}/prediction-logs/{log_id}", timeout=API_TIMEOUT)
    response.raise_for_status()
    return response.json()


def delete_all_prediction_logs() -> dict:
    """Xoá toàn bộ lịch sử nhận diện."""
    response = requests.delete(f"{API_BASE_URL}/prediction-logs", timeout=API_TIMEOUT)
    response.raise_for_status()
    return response.json()