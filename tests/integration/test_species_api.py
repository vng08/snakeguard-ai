from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.app.api import species as species_api
from backend.app.api.dependencies import get_db


class FakeSpecies:
    """Giả lập model SnakeSpecies trả về từ database."""

    def __init__(self, id, binomial_name, vietnamese_name, family, genus, is_mivs):
        self.id = id
        self.binomial_name = binomial_name
        self.vietnamese_name = vietnamese_name
        self.family = family
        self.genus = genus
        self.is_mivs = is_mivs


class FakeQuery:
    """Giả lập SQLAlchemy query."""

    def __init__(self, species):
        self.species = species

    def all(self):
        return self.species


class FakeDB:
    """Giả lập database session."""

    def __init__(self, species):
        self.species = species

    def query(self, *args, **kwargs):
        return FakeQuery(self.species)


def create_client(species):
    """Tạo FastAPI test client với database giả."""
    app = FastAPI()
    app.include_router(species_api.router)
    app.dependency_overrides[get_db] = lambda: FakeDB(species)
    return TestClient(app)


def test_get_species():
    species = [
        FakeSpecies(
            id=1,
            binomial_name="Naja kaouthia",
            vietnamese_name="Rắn hổ mang một mắt kính",
            family="Elapidae",
            genus="Naja",
            is_mivs=True,
        ),
        FakeSpecies(
            id=2,
            binomial_name="Ptyas korros",
            vietnamese_name="Rắn ráo",
            family="Colubridae",
            genus="Ptyas",
            is_mivs=False,
        ),
    ]

    client = create_client(species)
    response = client.get("/species")

    assert response.status_code == 200

    data = response.json()
    assert len(data) == 2
    assert data[0]["binomial_name"] == "Naja kaouthia"
    assert data[0]["is_mivs"] is True
    assert data[1]["binomial_name"] == "Ptyas korros"
    assert data[1]["is_mivs"] is False


def test_get_species_empty():
    client = create_client([])
    response = client.get("/species")

    assert response.status_code == 200
    assert response.json() == []