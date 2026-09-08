from io import BytesIO
from PIL import Image, UnidentifiedImageError
from backend.app.core.config import settings


def decode_image(data: bytes) -> Image.Image:
    """Đọc và validate ảnh đầu vào trước khi đưa vào pipeline."""
    if not data:
        raise ValueError("Empty image file.")

    if len(data) > settings.MAX_IMAGE_SIZE:
        raise ValueError("Image exceeds 10 MB.")

    try:
        image = Image.open(BytesIO(data))
        image.load()
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError("Invalid image file.") from exc

    if image.width * image.height > settings.MAX_IMAGE_PIXELS:
        raise ValueError("Image dimensions are too large.")

    return image.convert("RGB")