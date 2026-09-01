from contextlib import asynccontextmanager
from fastapi import FastAPI
from ml.pipeline import SnakeGuardPipeline
from backend.app.core.config import settings

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.pipeline = SnakeGuardPipeline(
        detector_path=settings.DETECTOR_MODEL_PATH,
        classifier_path=settings.CLASSIFIER_MODEL_PATH,
        classes_path=settings.CLASSES_PATH,
    )
    yield
    app.state.pipeline = None