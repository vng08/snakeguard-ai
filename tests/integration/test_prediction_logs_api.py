from datetime import datetime

from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.app.api import prediction_logs as prediction_logs_api
from backend.app.api.dependencies import get_db


class FakeDB:
    """Giả lập database session cho API test."""
    pass


def create_client():
    """Tạo FastAPI test client với database giả."""
    app = FastAPI()
    app.include_router(prediction_logs_api.router)
    app.dependency_overrides[get_db] = lambda: FakeDB()
    return TestClient(app)


def make_log():
    """Tạo prediction log giả dùng cho test."""
    return {
        "id": 1,
        "image_url": "test.jpg",
        "bbox": [10, 20, 80, 90],
        "detection_confidence": 0.95,
        "predicted_species_id": 1,
        "confidence": 0.90,
        "top_k_predictions": [
            {
                "label_idx": 0,
                "class_id": 1,
                "binomial_name": "Naja kaouthia",
                "confidence": 0.90,
                "MIVS": 1,
            }
        ],
        "model_version": "test-model",
        "created_at": datetime(2026, 9, 7, 12, 0, 0),
    }


def test_get_prediction_logs(monkeypatch):
    """Kiểm tra lấy toàn bộ prediction logs."""
    monkeypatch.setattr(prediction_logs_api, "get_prediction_logs", lambda db: [make_log()])

    client = create_client()
    response = client.get("/prediction-logs")

    assert response.status_code == 200

    data = response.json()
    assert len(data) == 1
    assert data[0]["id"] == 1
    assert data[0]["predicted_species_id"] == 1
    assert data[0]["confidence"] == 0.90


def test_delete_prediction_log(monkeypatch):
    """Kiểm tra xoá một prediction log thành công."""
    monkeypatch.setattr(prediction_logs_api, "delete_prediction_log", lambda db, log_id: True)

    client = create_client()
    response = client.delete("/prediction-logs/1")

    assert response.status_code == 200
    assert response.json()["id"] == 1


def test_delete_prediction_log_not_found(monkeypatch):
    """Kiểm tra xoá prediction log không tồn tại."""
    monkeypatch.setattr(prediction_logs_api, "delete_prediction_log", lambda db, log_id: False)

    client = create_client()
    response = client.delete("/prediction-logs/999")

    assert response.status_code == 404
    assert response.json()["detail"] == "Prediction log not found."


def test_delete_all_prediction_logs(monkeypatch):
    """Kiểm tra xoá toàn bộ prediction logs."""
    monkeypatch.setattr(prediction_logs_api, "delete_all_prediction_logs", lambda db: 5)

    client = create_client()
    response = client.delete("/prediction-logs")

    assert response.status_code == 200
    assert response.json()["deleted_count"] == 5