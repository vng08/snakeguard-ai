from io import BytesIO

from fastapi import FastAPI
from fastapi.testclient import TestClient
from PIL import Image

from backend.app.api import predict as predict_api
from backend.app.api.dependencies import get_db
from backend.app.services.prediction import prediction_service


class FakeQuery:
    """Giả lập SQLAlchemy query dùng trong PredictionService."""

    def filter(self, *args, **kwargs):
        return self

    def first(self):
        return None


class FakeDB:
    """Giả lập database session."""

    def query(self, *args, **kwargs):
        return FakeQuery()


class MockPipeline:
    """Giả lập pipeline nhận diện rắn."""

    def __init__(self, results):
        self.results = results

    def predict(self, image, top_k=3):
        return self.results


def make_image_bytes():
    """Tạo ảnh JPEG hợp lệ dùng cho test."""
    buffer = BytesIO()
    Image.new("RGB", (100, 100)).save(buffer, format="JPEG")
    return buffer.getvalue()


def create_client(monkeypatch, results):
    """Tạo FastAPI test client với pipeline và database giả."""
    app = FastAPI()
    app.include_router(predict_api.router)
    app.state.pipeline = MockPipeline(results)
    app.dependency_overrides[get_db] = lambda: FakeDB()

    # Không lưu ảnh và prediction log thật khi chạy test
    monkeypatch.setattr(prediction_service, "save_prediction_image", lambda image: "test.jpg")
    monkeypatch.setattr(prediction_service, "save_prediction_log", lambda *args, **kwargs: None)

    return TestClient(app)


def test_predict_with_snake(monkeypatch):
    results = [{
        "bbox": [10, 20, 80, 90],
        "detection_confidence": 0.95,
        "predictions": [
            {"label_idx": 0, "class_id": 1, "binomial_name": "Test snake", "confidence": 0.90, "MIVS": 0},
            {"label_idx": 1, "class_id": 2, "binomial_name": "Snake B", "confidence": 0.07, "MIVS": 0},
            {"label_idx": 2, "class_id": 3, "binomial_name": "Snake C", "confidence": 0.03, "MIVS": 1},
        ],
    }]

    client = create_client(monkeypatch, results)
    response = client.post("/predict", files={"file": ("snake.jpg", make_image_bytes(), "image/jpeg")})

    assert response.status_code == 200
    assert response.json()["detected"] is True
    assert response.json()["detection"]["bbox"] == [10, 20, 80, 90]
    assert len(response.json()["predictions"]) == 3


def test_predict_without_snake(monkeypatch):
    client = create_client(monkeypatch, [])
    response = client.post("/predict", files={"file": ("image.jpg", make_image_bytes(), "image/jpeg")})

    assert response.status_code == 200
    assert response.json()["detected"] is False
    assert response.json()["detection"] is None
    assert response.json()["predictions"] == []


def test_invalid_mime_type(monkeypatch):
    client = create_client(monkeypatch, [])
    response = client.post("/predict", files={"file": ("test.txt", b"hello", "text/plain")})

    assert response.status_code == 415


def test_corrupted_image(monkeypatch):
    client = create_client(monkeypatch, [])
    response = client.post("/predict", files={"file": ("broken.jpg", b"not-an-image", "image/jpeg")})

    assert response.status_code == 400