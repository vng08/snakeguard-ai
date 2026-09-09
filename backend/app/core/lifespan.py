from contextlib import asynccontextmanager

from fastapi import FastAPI

from backend.app.core.config import settings
from backend.app.services.rag.embedding_service import get_embedding_model
from backend.app.services.rag.tools.reranker import get_reranker_model
from ml.pipeline import SnakeGuardPipeline


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Khởi tạo các model dùng chung khi FastAPI startup."""
    app.state.pipeline = SnakeGuardPipeline(
        detector_path=settings.DETECTOR_MODEL_PATH,
        classifier_path=settings.CLASSIFIER_MODEL_PATH,
        classes_path=settings.CLASSES_PATH,
    )

    # Preload model RAG để tránh cold-start ở request đầu tiên
    get_embedding_model()
    get_reranker_model()

    yield

    app.state.pipeline = None