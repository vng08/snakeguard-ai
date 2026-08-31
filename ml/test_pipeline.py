from pathlib import Path

from ml.pipeline import SnakeGuardPipeline

ROOT_DIR = Path(__file__).resolve().parents[1]

image_path = ROOT_DIR / "data" / "samples" / "Achalinus-rufescens-1.jpg"
detector_path = ROOT_DIR / "models" / "detection" / "yolo26n_best.pt"
classifier_path = ROOT_DIR / "models" / "classification" / "convnextv2_best.pt"
classes_path = ROOT_DIR / "data" / "processed" / "classes.csv"

pipeline = SnakeGuardPipeline(
    detector_path=detector_path,
    classifier_path=classifier_path,
    classes_path=classes_path,
)

results = pipeline.predict(image_path, top_k=3)

if not results:
    print("No snake detected.")

for i, result in enumerate(results, 1):
    print(f"\nDetection {i}")
    print(f"BBox: {result['bbox']}")
    print(f"Detection confidence: {result['detection_confidence']:.4f}")

    for rank, pred in enumerate(result["predictions"], 1):
        print(
            f"Top {rank}: {pred['binomial_name']} | "
            f"Confidence: {pred['confidence']:.4f} | "
            f"MIVS: {pred['MIVS']}"
        )