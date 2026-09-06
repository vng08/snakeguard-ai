from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT_DIR = Path(__file__).resolve().parents[3]

class Settings(BaseSettings):
    DATABASE_URL: str
    DETECTOR_MODEL_PATH: Path = ROOT_DIR / "models/detection/yolo26n_best.pt"
    CLASSIFIER_MODEL_PATH: Path = ROOT_DIR / "models/classification/convnextv2_best.pt"
    CLASSES_PATH: Path = ROOT_DIR / "data/processed/classes.csv"
    PREDICTION_STORAGE_DIR: Path = ROOT_DIR / "data/predictions"

    EMBEDDING_MODEL_NAME: str = "BAAI/bge-m3"
    RERANKER_MODEL_NAME: str = "BAAI/bge-reranker-v2-m3"
    EMBEDDING_DIM: int = 1024

    GROQ_API_KEY: str
    TAVILY_API_KEY: str
    LLM_MODEL: str = "openai/gpt-oss-20b"

    IMAGE_RETRIEVAL_MODEL_NAME: str = "google/siglip2-base-patch16-224"
    IMAGE_RETRIEVAL_EMBEDDING_DIM: int = 768

    MODEL_VERSION: str = "yolo26n_best+convnextv2_best"

    model_config = SettingsConfigDict(env_file=".env")

settings = Settings()