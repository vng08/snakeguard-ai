from io import BytesIO

from PIL import Image, ImageDraw, ImageOps


def _transform_bbox(bbox: list[int], width: int, height: int, orientation: int) -> list[int]:
    """Biến đổi bbox theo cùng EXIF orientation của ảnh."""
    x1, y1, x2, y2 = bbox
    points = [(x1, y1), (x2, y1), (x1, y2), (x2, y2)]

    def transform(x, y):
        if orientation == 2:
            return width - x, y
        if orientation == 3:
            return width - x, height - y
        if orientation == 4:
            return x, height - y
        if orientation == 5:
            return y, x
        if orientation == 6:
            return height - y, x
        if orientation == 7:
            return height - y, width - x
        if orientation == 8:
            return y, width - x
        return x, y

    transformed = [transform(x, y) for x, y in points]
    xs = [point[0] for point in transformed]
    ys = [point[1] for point in transformed]

    return [int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))]


def draw_detection_box(image_bytes: bytes, bbox: list[int], expand_ratio: float = 0.1) -> Image.Image:
    """Xoay ảnh và bbox theo EXIF rồi vẽ bounding box."""
    image = Image.open(BytesIO(image_bytes))
    width, height = image.size
    orientation = image.getexif().get(274, 1)

    # Bbox từ model đang thuộc hệ tọa độ ảnh raw
    bbox = _transform_bbox(bbox, width, height, orientation)

    # Xoay ảnh về đúng hướng hiển thị
    image = ImageOps.exif_transpose(image).convert("RGB")

    x1, y1, x2, y2 = bbox
    box_width, box_height = x2 - x1, y2 - y1
    padding_x, padding_y = box_width * expand_ratio / 2, box_height * expand_ratio / 2

    x1 = max(0, int(x1 - padding_x))
    y1 = max(0, int(y1 - padding_y))
    x2 = min(image.width, int(x2 + padding_x))
    y2 = min(image.height, int(y2 + padding_y))

    ImageDraw.Draw(image).rectangle([x1, y1, x2, y2], outline="red", width=4)
    return image