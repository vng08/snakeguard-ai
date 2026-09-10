from sentence_transformers import SentenceTransformer

from backend.app.core.config import settings

_embedding_model = None


def get_embedding_model() -> SentenceTransformer:
    """Lazy-load embedding model và dùng chung trong toàn bộ RAG."""
    global _embedding_model

    if _embedding_model is None:
        _embedding_model = SentenceTransformer(settings.EMBEDDING_MODEL_NAME)

    return _embedding_model


def embed_query(text: str) -> list[float]:
    """Embedding query thành vector đã normalize."""
    model = get_embedding_model()
    embedding = model.encode(text, normalize_embeddings=True, convert_to_numpy=True)
    return embedding.tolist()