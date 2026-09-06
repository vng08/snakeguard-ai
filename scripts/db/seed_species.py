import pandas as pd
from backend.app.db.models import SnakeSpecies
from backend.app.db.session import SessionLocal

DATA_PATH = "data/processed/species_metadata.csv"


def seed_species():
    df = pd.read_csv(DATA_PATH)
    db = SessionLocal()

    try:
        for row in df.itertuples(index=False):
            db.add(
                SnakeSpecies(
                    binomial_name=row.binomial_name, vietnamese_name=row.vietnamese_name,
                    family=row.family, genus=row.genus, is_mivs=bool(row.MIVS),
                )
            )

        db.commit()
        print(f"Seeded {len(df)} species.")

    except Exception:
        db.rollback()
        raise

    finally:
        db.close()


if __name__ == "__main__":
    seed_species()