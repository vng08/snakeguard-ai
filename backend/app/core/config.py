from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT_DIR = Path(__file__).resolve().parents[3]

class Settings(BaseSettings):
    DATABASE_URL: str
    DETECTOR_MODEL_PATH: Path = ROOT_DIR / "models/detection/yolo26n_best.pt"
    CLASSIFIER_MODEL_PATH: Path = ROOT_DIR / "models/classification/convnextv2_best.pt"
    CLASSES_PATH: Path = ROOT_DIR / "data/processed/classes.csv"
    PREDICTION_STORAGE_DIR: Path = ROOT_DIR / "data/predictions"
    MODEL_VERSION: str = "yolo26n_best+convnextv2_best"

    model_config = SettingsConfigDict(env_file=".env")

settings = Settings()