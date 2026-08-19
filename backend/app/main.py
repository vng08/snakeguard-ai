from fastapi import FastAPI

from backend.app.api.species import router as species_router


app = FastAPI()

app.include_router(species_router, prefix="/api")


@app.get("/health", tags=["Health"])
def health_check():
    return {"status": "ok"}