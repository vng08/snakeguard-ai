from ml.detection.inference import SnakeDetector
from ml.classification.inference import SnakeClassifier


class SnakeGuardPipeline:
    def __init__(self, detector_path, classifier_path, classes_path, detector_conf=0.48, detector_imgsz=640):
        self.detector = SnakeDetector(
            model_path=detector_path,
            conf=detector_conf,
            imgsz=detector_imgsz,
        )

        self.classifier = SnakeClassifier(
            model_path=classifier_path,
            classes_path=classes_path,
        )

    def predict(self, image_path, top_k=3):
        detections = self.detector.predict(image_path)
        results = []

        for detection in detections:
            predictions = self.classifier.predict(detection["crop"], top_k=top_k)

            results.append({
                "bbox": detection["bbox"],
                "detection_confidence": detection["confidence"],
                "predictions": predictions,
            })

        return results