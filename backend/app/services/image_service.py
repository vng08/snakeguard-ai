from io import BytesIO
from PIL import Image, UnidentifiedImageError

MAX_IMAGE_SIZE = 10 * 1024 * 1024
MAX_IMAGE_PIXELS = 20_000_000

def decode_image(data: bytes) -> Image.Image:
    if not data: raise ValueError("Empty image file.")
    if len(data) > MAX_IMAGE_SIZE: raise ValueError("Image exceeds 10 MB.")

    try:
        image = Image.open(BytesIO(data))
        image.load()
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError("Invalid image file.") from exc

    if image.width * image.height > MAX_IMAGE_PIXELS:
        raise ValueError("Image dimensions are too large.")

    return image.convert("RGB")