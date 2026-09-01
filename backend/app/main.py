from fastapi import FastAPI
from backend.app.api.species import router as species_router
from backend.app.api.predict import router as predict_router
from backend.app.core.lifespan import lifespan

app = FastAPI(lifespan=lifespan)

app.include_router(species_router, prefix="/api")
app.include_router(predict_router, prefix="/api")

@app.get("/health", tags=["Health"])
def health_check():
    return {"status": "ok"}