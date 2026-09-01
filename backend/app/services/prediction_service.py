from PIL import Image
from ml.pipeline import SnakeGuardPipeline

class PredictionService:
    def __init__(self, pipeline: SnakeGuardPipeline):
        self.pipeline = pipeline

    def predict(self, image: Image.Image, top_k: int = 3):
        return self.pipeline.predict(image, top_k=top_k)