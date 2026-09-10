from pathlib import Path

import pandas as pd
import timm
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps
from torch import nn
from torchvision import transforms
from torchvision.transforms import InterpolationMode


class PadToSquare:
    def __call__(self, image):
        w, h = image.size
        size = max(w, h)
        pad_w, pad_h = size - w, size - h
        padding = (pad_w // 2, pad_h // 2, pad_w - pad_w // 2, pad_h - pad_h // 2)
        return ImageOps.expand(image, padding, fill=(124, 116, 104))


class ArcFace(nn.Module):
    def __init__(self, in_features, num_classes, scale=30.0, margin=0.3):
        super().__init__()
        self.scale = scale
        self.margin = margin
        self.weight = nn.Parameter(torch.empty(num_classes, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, features, labels=None):
        cosine = F.linear(F.normalize(features), F.normalize(self.weight))
        cosine = cosine.clamp(-1 + 1e-7, 1 - 1e-7)

        if labels is None:
            return cosine * self.scale

        theta = torch.acos(cosine)
        target_cosine = torch.cos(theta + self.margin)
        one_hot = F.one_hot(labels, num_classes=cosine.size(1)).float()
        logits = cosine * (1 - one_hot) + target_cosine * one_hot
        return logits * self.scale


class SnakeClassifier:
    def __init__(self, model_path, classes_path, img_size=384, device=None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.img_size = img_size

        self.classes = pd.read_csv(classes_path).sort_values("label_idx").reset_index(drop=True)
        self.num_classes = len(self.classes)

        assert self.classes["label_idx"].tolist() == list(range(self.num_classes))

        self.transform = transforms.Compose([
            PadToSquare(),
            transforms.Resize((img_size, img_size), interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

        self.model = timm.create_model(
            "convnextv2_base.fcmae_ft_in22k_in1k_384",
            pretrained=False,
            num_classes=0,
            drop_path_rate=0.2,
        )

        self.arcface = ArcFace(
            in_features=self.model.num_features,
            num_classes=self.num_classes,
            scale=30.0,
            margin=0.3,
        )

        checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
        self.model.load_state_dict(checkpoint["model"], strict=True)
        self.arcface.load_state_dict(checkpoint["arcface"], strict=True)

        self.model = self.model.to(self.device).eval()
        self.arcface = self.arcface.to(self.device).eval()

    def predict(self, image, top_k=3):
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert("RGB")
        else:
            image = image.convert("RGB")

        tensor = self.transform(image).unsqueeze(0).to(self.device)

        with torch.inference_mode():
            with torch.autocast(device_type=self.device.type, enabled=self.device.type == "cuda"):
                features = self.model(tensor)
                logits = self.arcface(features)

            probs = torch.softmax(logits, dim=1)
            confidences, indices = probs.topk(top_k, dim=1)

        predictions = []

        for idx, confidence in zip(indices[0].cpu().tolist(), confidences[0].cpu().tolist()):
            species = self.classes.iloc[idx]

            predictions.append({
                "label_idx": int(species["label_idx"]),
                "class_id": int(species["class_id"]),
                "binomial_name": species["binomial_name"],
                "confidence": float(confidence),
                "MIVS": int(species["MIVS"]),
            })

        return predictions