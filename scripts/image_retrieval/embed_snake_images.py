from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR))

from PIL import Image

from backend.app.db.models import SnakeImage
from backend.app.db.session import SessionLocal


def resolve_image_path(image_url: str) -> Path:
    """Chuyển image_url trong DB thành đường dẫn file thực tế."""
    path = Path(image_url)
    return path if path.is_absolute() else ROOT_DIR / path


def index_embeddings():
    """Tạo embedding chỉ cho các ảnh chưa được index."""
    db = SessionLocal()

    try:
        # Chỉ lấy những ảnh chưa có embedding
        images = db.query(SnakeImage).filter(
            SnakeImage.embedding.is_(None)
        ).all()

        if not images:
            print("No images need embedding. Skipping SigLIP loading.")
            return

        print(f"Images to embed: {len(images)}")

        # Chỉ load model khi thực sự còn ảnh cần embedding
        from ml.image_retrieval.siglip_encoder import SigLIPEncoder
        encoder = SigLIPEncoder()

        indexed = 0
        failed = 0

        for image_row in images:
            try:
                image_path = resolve_image_path(image_row.image_url)

                with Image.open(image_path) as image:
                    image = image.convert("RGB")
                    image_row.embedding = encoder.encode_image(image)

                indexed += 1

                # Commit theo batch để tránh giữ transaction quá lớn
                if indexed % 100 == 0:
                    db.commit()
                    print(f"Indexed: {indexed}")

            except Exception as e:
                failed += 1
                print(f"Failed image_id={image_row.id}: {e}")

        db.commit()

        print(f"Indexed: {indexed}")
        print(f"Failed : {failed}")

    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


if __name__ == "__main__":
    index_embeddings()