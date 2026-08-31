from pathlib import Path

from ml.classification.inference import SnakeClassifier

ROOT_DIR = Path(__file__).resolve().parents[2]

image_path = ROOT_DIR / "data" / "samples" / "Achalinus rufescens.jpg"
model_path = ROOT_DIR / "models" / "classification" / "convnextv2_best.pt"
classes_path = ROOT_DIR / "data" / "processed" / "classes.csv"

classifier = SnakeClassifier(
    model_path=model_path,
    classes_path=classes_path,
)

predictions = classifier.predict(image_path, top_k=3)

for pred in predictions:
    print(pred)