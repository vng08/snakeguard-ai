from uuid import uuid4
from PIL import Image
from backend.app.core.config import ROOT_DIR, settings

def save_prediction_image(image: Image.Image) -> str:
    storage_dir = settings.PREDICTION_STORAGE_DIR
    storage_dir.mkdir(parents=True, exist_ok=True)

    path = storage_dir / f"{uuid4().hex}.jpg"
    image.save(path, format="JPEG", quality=95)
    return str(path.relative_to(ROOT_DIR))