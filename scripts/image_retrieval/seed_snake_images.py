from pathlib import Path
import random
import sys

import pandas as pd
from sqlalchemy import select

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR))

from backend.app.db.models import SnakeImage, SnakeSpecies
from backend.app.db.session import SessionLocal

CLASSES_PATH = ROOT_DIR / "data/processed/classes.csv"
CROP_DIR = ROOT_DIR / "dataset/yolo-crop/train"
IMAGES_PER_SPECIES = 15

random.seed(42)


def seed_snake_images():
    """Seed ảnh tham chiếu cho từng loài, bỏ qua ảnh đã tồn tại trong DB."""
    df = pd.read_csv(CLASSES_PATH)
    label_map = df.set_index("label_idx")["binomial_name"].to_dict()

    db = SessionLocal()
    inserted = 0
    skipped = 0

    try:
        # Lấy trước các image_url đã tồn tại để tránh duplicate
        existing_urls = set(db.scalars(select(SnakeImage.image_url)).all())

        for class_dir in sorted(CROP_DIR.iterdir()):
            if not class_dir.is_dir():
                continue

            label_idx = int(class_dir.name)
            binomial_name = label_map.get(label_idx)

            if not binomial_name:
                print(f"No mapping for label {label_idx}")
                continue

            species = db.query(SnakeSpecies).filter(
                SnakeSpecies.binomial_name == binomial_name
            ).first()

            if species is None:
                print(f"Species not found: {binomial_name}")
                continue

            images = sorted(
                p for p in class_dir.iterdir()
                if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
            )
            images = random.sample(images, min(IMAGES_PER_SPECIES, len(images)))

            for image_path in images:
                relative_path = str(image_path.relative_to(ROOT_DIR))

                if relative_path in existing_urls:
                    skipped += 1
                    continue

                db.add(
                    SnakeImage(
                        species_id=species.id,
                        image_url=relative_path,
                        source="SnakeCLEF2023",
                        license=None,
                    )
                )

                existing_urls.add(relative_path)
                inserted += 1

        db.commit()

        print(f"Inserted: {inserted}")
        print(f"Skipped : {skipped}")

    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


if __name__ == "__main__":
    seed_snake_images()