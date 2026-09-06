from pathlib import Path
import shutil
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
META_DIR = ROOT / "data" / "processed"
TRAIN_MAIN = ROOT / "dataset" / "train-main"
TRAIN_HMP = ROOT / "dataset" / "train-HMP"
TEST_SOURCE = ROOT / "dataset" / "val"
OUTPUT_DIR = ROOT / "dataset" / "selected"


def get_source(image_path: str, split: str) -> Path:
    # Test split comes from official validation set
    if split == "test":
        return TEST_SOURCE / image_path

    # HMP images use a different root directory
    if image_path.startswith("HMP/"):
        return TRAIN_HMP / image_path.removeprefix("HMP/")

    return TRAIN_MAIN / image_path


def select_split(split: str):
    csv_path = META_DIR / f"{split}.csv"
    df = pd.read_csv(csv_path)

    success = 0
    missing = 0

    for _, row in df.iterrows():
        image_path = str(row["image_path"])
        label_idx = int(row["label_idx"])
        src = get_source(image_path, split)

        # Class folders: 000 ... 108
        class_dir = OUTPUT_DIR / split / f"{label_idx:03d}"
        class_dir.mkdir(parents=True, exist_ok=True)
        dst = class_dir / Path(image_path).name

        if src.exists():
            shutil.copy2(src, dst)
            success += 1
        else:
            missing += 1
            print(f"[MISSING] {image_path} -> {src}")

    print(f"{split:<5} | " f"total: {len(df):>5} | " f"success: {success:>5} | " f"missing: {missing:>5}")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for split in ["train", "val", "test"]:
        select_split(split)


if __name__ == "__main__":
    main()