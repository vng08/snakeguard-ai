from fastapi.testclient import TestClient

from backend.app.api.dependencies import get_db
from backend.app.main import app


def fake_get_db():
    yield None


app.dependency_overrides[get_db] = fake_get_db
client = TestClient(app)


# Test mô tả rắn hợp lệ -> API phải trả Top-K species
def test_search_by_description_success(monkeypatch):
    monkeypatch.setattr("backend.app.api.search.classify_description", lambda description: "snake")

    monkeypatch.setattr(
        "backend.app.api.search.hybrid_search",
        lambda db, description, top_k: [{
            "species_id": 1,
            "binomial_name": "Ahaetulla prasina",
            "vietnamese_name": "Rắn roi thường",
            "image_url": "dataset/sample.jpg",
            "image_score_normalized": 1.0,
            "knowledge_score_normalized": 0.8,
            "hybrid_score": 0.9,
            "is_mivs": False,
        }],
    )

    response = client.post(
        "/search-by-description",
        json={"description": "rắn màu xanh, thân mảnh, sống trên cây", "top_k": 5},
    )

    assert response.status_code == 200

    data = response.json()
    assert data["description"] == "rắn màu xanh, thân mảnh, sống trên cây"
    assert len(data["results"]) == 1
    assert data["results"][0]["binomial_name"] == "Ahaetulla prasina"


# Test nội dung không phải mô tả rắn -> API phải từ chối retrieval
def test_search_by_description_unspecified(monkeypatch):
    monkeypatch.setattr("backend.app.api.search.classify_description", lambda description: "unspecified")

    response = client.post(
        "/search-by-description",
        json={"description": "hôm nay tôi muốn ăn phở", "top_k": 5},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == ("Nội dung không chứa đủ đặc điểm để nhận dạng rắn. Hãy cung cấp chính xác thông tin nhận diện.")


# Test description quá ngắn -> Pydantic phải reject request
def test_search_by_description_invalid_description():
    response = client.post(
        "/search-by-description",
        json={"description": "a", "top_k": 5},
    )

    assert response.status_code == 422