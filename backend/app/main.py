from fastapi import FastAPI
from backend.app.api.species import router as species_router
from backend.app.api.predict import router as predict_router
from backend.app.core.lifespan import lifespan
from backend.app.api.chat import router as chat_router

app = FastAPI(lifespan=lifespan)

app.include_router(species_router, prefix="/api")
app.include_router(predict_router, prefix="/api")
app.include_router(chat_router)

@app.get("/health", tags=["Health"])
def health_check():
    return {"status": "ok"}