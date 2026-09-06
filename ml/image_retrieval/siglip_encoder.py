from PIL import Image
import torch
from transformers import AutoModel, AutoProcessor

from backend.app.core.config import settings


class SigLIPEncoder:
    def __init__(self):
        self.processor = AutoProcessor.from_pretrained(settings.IMAGE_RETRIEVAL_MODEL_NAME)
        self.model = AutoModel.from_pretrained(settings.IMAGE_RETRIEVAL_MODEL_NAME)
        self.model.eval()

    def encode_text(self, text: str) -> list[float]:
        inputs = self.processor(text=[text], return_tensors="pt", padding="max_length")

        with torch.no_grad():
            outputs = self.model.get_text_features(**inputs)
            embedding = outputs.pooler_output

        embedding = torch.nn.functional.normalize(embedding, dim=-1)
        return embedding[0].cpu().tolist()

    def encode_image(self, image: Image.Image) -> list[float]:
        inputs = self.processor(images=image, return_tensors="pt")

        with torch.no_grad():
            outputs = self.model.get_image_features(**inputs)
            embedding = outputs.pooler_output

        embedding = torch.nn.functional.normalize(embedding, dim=-1)
        return embedding[0].cpu().tolist()