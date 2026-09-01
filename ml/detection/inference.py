from pathlib import Path
from PIL import Image
from ultralytics import YOLO


class SnakeDetector:
    def __init__(self, model_path, conf=0.48, imgsz=640): # Confidence threshold selected by maximizing F1-score.
        self.model = YOLO(model_path)
        self.conf = conf
        self.imgsz = imgsz

    def predict(self, image):
        if isinstance(image, (str, Path)): image = Image.open(image).convert("RGB")
        else: image = image.convert("RGB")

        result = self.model.predict(source=image, conf=self.conf, imgsz=self.imgsz, verbose=False)[0]
        detections = []

        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            confidence = float(box.conf[0])
            detections.append({"bbox": [x1, y1, x2, y2], "confidence": confidence, "crop": image.crop((x1, y1, x2, y2))})

        return detections