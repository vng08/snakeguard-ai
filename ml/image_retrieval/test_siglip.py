from pathlib import Path

from PIL import Image

from ml.image_retrieval.siglip_encoder import SigLIPEncoder

ROOT_DIR = Path(__file__).resolve().parents[2]

encoder = SigLIPEncoder()

image = Image.open(ROOT_DIR / "data/samples/ho-mang.jpg").convert("RGB")

image_embedding = encoder.encode_image(image)
text_embedding = encoder.encode_text("rắn hổ mang màu đen, đầu bè")

print("Image embedding length:", len(image_embedding))
print("Text embedding length:", len(text_embedding))