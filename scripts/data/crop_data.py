from pathlib import Path
import pandas as pd

from ml.detection.inference import SnakeDetector


ROOT = Path(__file__).resolve().parents[1]
INPUT_DIR = ROOT / "dataset" / "selected"
OUTPUT_DIR = ROOT / "dataset" / "yolo-crop"
MODEL_PATH = ROOT / "models" / "detection" / "yolo26n_best.pt"
REPORT_PATH = OUTPUT_DIR / "crop_report.csv"


def process_split(split: str, detector: SnakeDetector, report: list):
    input_dir = INPUT_DIR / split
    output_dir = OUTPUT_DIR / split

    total = cropped = no_detection = multiple = invalid = 0

    for class_dir in sorted(input_dir.iterdir()):
        if not class_dir.is_dir():
            continue

        output_class_dir = output_dir / class_dir.name
        output_class_dir.mkdir(parents=True, exist_ok=True)

        for image_path in class_dir.iterdir():
            if not image_path.is_file():
                continue

            total += 1

            base_report = {
                "split": split,
                "filepath": str(image_path.relative_to(ROOT)),
                "num_detections": 0,
                "confidence": None,
            }

            try:
                detections = detector.predict(image_path)

            # Invalid/corrupted image or inference failure
            except Exception:
                invalid += 1
                report.append({**base_report, "status": "invalid_image"})
                continue

            # YOLO could not detect any snake
            if not detections:
                no_detection += 1
                report.append({**base_report, "status": "no_detection"})
                continue

            # Use the highest-confidence detection
            best = max(detections, key=lambda x: x["confidence"])
            best["crop"].save(output_class_dir / image_path.name)
            cropped += 1

            # Multiple snakes/detections found
            if len(detections) > 1:
                multiple += 1
                report.append({
                    **base_report,
                    "status": "multiple_detections",
                    "num_detections": len(detections),
                    "confidence": best["confidence"],
                })

    print(
        f"{split:<5} | total: {total:>5} | cropped: {cropped:>5} | "
        f"no_det: {no_detection:>5} | multi: {multiple:>5} | invalid: {invalid:>5}"
    )


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    detector = SnakeDetector(MODEL_PATH, conf=0.2, imgsz=640)
    report = []

    for split in ["train", "val", "test"]:
        process_split(split, detector, report)

    pd.DataFrame(report, columns=["split", "filepath", "status", "num_detections", "confidence"],).to_csv(REPORT_PATH, index=False)
    print(f"Report saved: {REPORT_PATH}")


if __name__ == "__main__":
    main()