from fastapi import FastAPI

from backend.app.api.species import router as species_router
from backend.app.db.base import Base
from backend.app.db.session import engine


app = FastAPI()

Base.metadata.create_all(bind=engine)

app.include_router(species_router, prefix="/api")


@app.get("/health", tags=["Health"])
def health_check():
    return {"status": "ok"}