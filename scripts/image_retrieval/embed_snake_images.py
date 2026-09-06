from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR))

from PIL import Image

from backend.app.db.models import SnakeImage
from backend.app.db.session import SessionLocal
from ml.image_retrieval.siglip_encoder import SigLIPEncoder


def resolve_image_path(image_url: str) -> Path:
    path = Path(image_url)
    return path if path.is_absolute() else ROOT_DIR / path


def index_embeddings():
    db = SessionLocal()
    encoder = SigLIPEncoder()
    indexed, skipped, failed = 0, 0, 0

    try:
        images = db.query(SnakeImage).all()

        for image_row in images:
            if image_row.embedding is not None:
                skipped += 1
                continue

            try:
                image_path = resolve_image_path(image_row.image_url)
                image = Image.open(image_path).convert("RGB")
                image_row.embedding = encoder.encode_image(image)
                indexed += 1

                if indexed % 100 == 0:
                    db.commit()
                    print(f"Indexed: {indexed}")

            except Exception as e:
                failed += 1
                print(f"Failed image_id={image_row.id}: {e}")

        db.commit()
        print(f"Indexed: {indexed}")
        print(f"Skipped: {skipped}")
        print(f"Failed: {failed}")

    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


if __name__ == "__main__":
    index_embeddings()