import pandas as pd
from sqlalchemy import select

from backend.app.db.models import SnakeSpecies
from backend.app.db.session import SessionLocal


DATA_PATH = "data/processed/species_metadata.csv"


def seed_species():
    """Seed species mới, bỏ qua species đã tồn tại."""
    df = pd.read_csv(DATA_PATH)
    db = SessionLocal()

    try:
        existing_names = set(db.scalars(select(SnakeSpecies.binomial_name)).all())
        inserted = 0

        for row in df.itertuples(index=False):
            # Bỏ qua species đã có để tránh duplicate khi chạy lại
            if row.binomial_name in existing_names:
                continue

            db.add(
                SnakeSpecies(
                    binomial_name=row.binomial_name,
                    vietnamese_name=row.vietnamese_name,
                    family=row.family,
                    genus=row.genus,
                    is_mivs=bool(row.MIVS),
                )
            )

            existing_names.add(row.binomial_name)
            inserted += 1

        db.commit()
        print(f"Seeded {inserted} new species.")

    except Exception:
        db.rollback()
        raise

    finally:
        db.close()


if __name__ == "__main__":
    seed_species()