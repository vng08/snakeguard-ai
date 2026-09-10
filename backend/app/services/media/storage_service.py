from uuid import uuid4

from PIL import Image

from backend.app.core.config import ROOT_DIR, settings


def save_prediction_image(image: Image.Image) -> str:
    """Lưu ảnh prediction và trả về đường dẫn tương đối."""
    storage_dir = settings.PREDICTION_STORAGE_DIR
    storage_dir.mkdir(parents=True, exist_ok=True)

    path = storage_dir / f"{uuid4().hex}.jpg"
    image.save(path, format="JPEG", quality=95)

    return str(path.relative_to(ROOT_DIR))


def delete_prediction_image(image_url: str | None) -> None:
    """Xoá một ảnh prediction theo đường dẫn đã lưu trong database."""
    if not image_url:
        return

    path = ROOT_DIR / image_url

    if path.exists():
        path.unlink()


def delete_all_prediction_images() -> None:
    """Xoá toàn bộ ảnh prediction đã lưu."""
    storage_dir = settings.PREDICTION_STORAGE_DIR

    if not storage_dir.exists():
        return

    for path in storage_dir.iterdir():
        if path.is_file():
            path.unlink()