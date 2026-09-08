from fastapi.testclient import TestClient

from backend.app.api.dependencies import get_db
from backend.app.main import app
from backend.app.schemas.search import DescriptionAnalysis


def fake_get_db():
    yield None


app.dependency_overrides[get_db] = fake_get_db
client = TestClient(app)


# Test mô tả rắn hợp lệ -> API phải trả Top-K species
def test_search_by_description_success(monkeypatch):
    monkeypatch.setattr(
        "backend.app.api.search.analyze_description",
        lambda description: DescriptionAnalysis(route="snake", focus="visual"),
    )

    monkeypatch.setattr(
        "backend.app.api.search.hybrid_search",
        lambda db, description, focus, search_mode, top_k: [{
            "species_id": 1,
            "binomial_name": "Ahaetulla prasina",
            "vietnamese_name": "Rắn roi thường",
            "image_url": "dataset/sample.jpg",
            "is_mivs": False,
            "score": 0.9,
        }],
    )

    response = client.post(
        "/search-by-description",
        json={
            "description": "rắn màu xanh, thân mảnh, sống trên cây",
            "search_mode": "fast",
            "top_k": 5,
        },
    )

    assert response.status_code == 200

    data = response.json()
    assert data["description"] == "rắn màu xanh, thân mảnh, sống trên cây"
    assert len(data["results"]) == 1
    assert data["results"][0]["binomial_name"] == "Ahaetulla prasina"
    assert data["results"][0]["score"] == 0.9


# Test nội dung không phải mô tả rắn -> API phải từ chối retrieval
def test_search_by_description_unspecified(monkeypatch):
    monkeypatch.setattr(
        "backend.app.api.search.analyze_description",
        lambda description: DescriptionAnalysis(route="unspecified", focus="mixed"),
    )

    response = client.post(
        "/search-by-description",
        json={"description": "hôm nay tôi muốn ăn phở", "search_mode": "fast", "top_k": 5},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == (
        "Nội dung không chứa đủ đặc điểm để nhận dạng rắn. Hãy cung cấp chính xác thông tin nhận diện."
    )


# Test description quá ngắn -> Pydantic phải reject request
def test_search_by_description_invalid_description():
    response = client.post(
        "/search-by-description",
        json={"description": "a", "search_mode": "fast", "top_k": 5},
    )

    assert response.status_code == 422