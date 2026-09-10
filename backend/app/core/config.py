from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT_DIR = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    # Cấu hình database
    DATABASE_URL: str

    # Cấu hình đường dẫn model và dữ liệu
    DETECTOR_MODEL_PATH: Path = ROOT_DIR / "models/detection/yolo26n_best.pt"
    CLASSIFIER_MODEL_PATH: Path = ROOT_DIR / "models/classification/convnextv2_best.pt"
    CLASSES_PATH: Path = ROOT_DIR / "data/processed/classes.csv"
    PREDICTION_STORAGE_DIR: Path = ROOT_DIR / "data/predictions"

    # Cấu hình ảnh đầu vào
    MAX_IMAGE_SIZE: int = 10 * 1024 * 1024
    MAX_IMAGE_PIXELS: int = 20_000_000
    ALLOWED_IMAGE_TYPES: set[str] = {"image/jpeg", "image/png", "image/webp", "image/bmp", "image/tiff"}

    # Cấu hình embedding và reranker cho RAG
    EMBEDDING_MODEL_NAME: str = "BAAI/bge-m3"
    RERANKER_MODEL_NAME: str = "BAAI/bge-reranker-v2-m3"
    EMBEDDING_DIM: int = 1024

    # Cấu hình Species Resolver
    SPECIES_FUZZY_THRESHOLD: float = 80
    SPECIES_AMBIGUITY_MARGIN: float = 5
    SPECIES_SEMANTIC_THRESHOLD: float = 0.65
    SPECIES_DEEP_CANDIDATE_K: int = 5
    SPECIES_DEEP_CONFIDENCE_THRESHOLD: float = 85
    SPECIES_RERANK_TOP_K: int = 5
    SPECIES_MATCH_WEIGHT: float = 0.6
    SPECIES_RERANK_WEIGHT: float = 0.4
    SPECIES_FINAL_THRESHOLD: float = 0.5
    SPECIES_FINAL_AMBIGUITY_MARGIN: float = 0.05

    # Cấu hình Fast và Deep retrieval
    RAG_FAST_TOP_K: int = 5
    RAG_DEEP_TOP_K: int = 5
    RAG_DEEP_CANDIDATE_K: int = 20

    # Cấu hình LLM
    LLM_PROVIDER: str
    LLM_API_KEY: str
    LLM_MODEL: str

    # Cấu hình web search
    TAVILY_API_KEY: str

    # Cấu hình model retrieval ảnh
    IMAGE_RETRIEVAL_MODEL_NAME: str = "google/siglip2-base-patch16-224"
    IMAGE_RETRIEVAL_EMBEDDING_DIM: int = 768

    # Cấu hình Search by Description
    SEARCH_FAST_CANDIDATE_K: int = 20
    SEARCH_DEEP_CANDIDATE_K: int = 30
    SEARCH_DEEP_FUSION_TOP_K: int = 15
    SEARCH_TOP_K: int = 5
    SEARCH_RRF_K: int = 20
    SEARCH_PROFILE_CHUNK_K: int = 6

    # Dynamic weight cho Weighted RRF
    SEARCH_VISUAL_IMAGE_WEIGHT: float = 0.7
    SEARCH_VISUAL_KNOWLEDGE_WEIGHT: float = 0.3
    SEARCH_MIXED_IMAGE_WEIGHT: float = 0.5
    SEARCH_MIXED_KNOWLEDGE_WEIGHT: float = 0.5
    SEARCH_CONTEXTUAL_IMAGE_WEIGHT: float = 0.4
    SEARCH_CONTEXTUAL_KNOWLEDGE_WEIGHT: float = 0.6

    # Boost nhẹ các loài MIVS
    SEARCH_MIVS_FACTOR: float = 1.03

    # Cấu hình Image Retrieval
    SEARCH_TOP_IMAGE_WEIGHTS: list[float] = [0.6, 0.3, 0.1]

    # Cấu hình version của pipeline prediction
    MODEL_VERSION: str = "yolo26n_best+convnextv2_best"

    model_config = SettingsConfigDict(env_file=".env")


settings = Settings()