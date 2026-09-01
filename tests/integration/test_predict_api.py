from io import BytesIO
from fastapi import FastAPI
from fastapi.testclient import TestClient
from PIL import Image

from backend.app.api import predict as predict_api
from backend.app.api.dependencies import get_db


class FakeQuery:
    def filter(self, *args, **kwargs): return self
    def first(self): return None


class FakeDB:
    def query(self, *args, **kwargs): return FakeQuery()


class MockPipeline:
    def __init__(self, results): self.results = results
    def predict(self, image, top_k=3): return self.results


def make_image_bytes():
    buffer = BytesIO()
    Image.new("RGB", (100, 100)).save(buffer, format="JPEG")
    return buffer.getvalue()


def create_client(monkeypatch, results):
    app = FastAPI()
    app.include_router(predict_api.router, prefix="/api")
    app.state.pipeline = MockPipeline(results)
    app.dependency_overrides[get_db] = lambda: FakeDB()

    monkeypatch.setattr(predict_api, "save_prediction_image", lambda image: "test.jpg")
    monkeypatch.setattr(predict_api, "save_prediction_log", lambda *args, **kwargs: None)
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
    response = client.post("/api/predict", files={"file": ("snake.jpg", make_image_bytes(), "image/jpeg")})

    assert response.status_code == 200
    assert response.json()["detected"] is True
    assert response.json()["detection"]["bbox"] == [10, 20, 80, 90]
    assert len(response.json()["predictions"]) == 3


def test_predict_without_snake(monkeypatch):
    client = create_client(monkeypatch, [])
    response = client.post("/api/predict", files={"file": ("image.jpg", make_image_bytes(), "image/jpeg")})

    assert response.status_code == 200
    assert response.json()["detected"] is False
    assert response.json()["detection"] is None
    assert response.json()["predictions"] == []


def test_invalid_mime_type(monkeypatch):
    client = create_client(monkeypatch, [])
    response = client.post("/api/predict", files={"file": ("test.txt", b"hello", "text/plain")})

    assert response.status_code == 415


def test_corrupted_image(monkeypatch):
    client = create_client(monkeypatch, [])
    response = client.post("/api/predict", files={"file": ("broken.jpg", b"not-an-image", "image/jpeg")})

    assert response.status_code == 400