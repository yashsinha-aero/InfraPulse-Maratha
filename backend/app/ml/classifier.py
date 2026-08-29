"""
Defect classifier: frozen ImageNet-pretrained MobileNetV2 backbone (generic,
COCO/ImageNet-scale pretraining is explicitly allowed) + a small custom
classification head trained by us on our own labeled defect dataset.

If `backbone_finetune.pt` exists it is loaded on top of the ImageNet backbone
(last-block fine-tuning done by app/ml/train.py) — still the same permitted
generic backbone, no defect/damage-specific pretrained checkpoint anywhere.
By default that file is absent and the backbone is pure frozen ImageNet.
"""
import os
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

CLASSES = ["spalling", "stagnant_water", "paint_peeling", "cracked_tiles"]
WEIGHTS_PATH = Path(__file__).resolve().parent.parent / "models_weights" / "head.pt"
BACKBONE_WEIGHTS_PATH = Path(__file__).resolve().parent.parent / "models_weights" / "backbone_finetune.pt"

_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


class DefectHead(nn.Module):
    """Small trainable head on top of frozen backbone features."""

    def __init__(self, in_features: int, num_classes: int = len(CLASSES)):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.net(x)


class DefectClassifier:
    def __init__(self):
        backbone = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
        self.feature_extractor = backbone.features
        if BACKBONE_WEIGHTS_PATH.exists():
            # last block fine-tuned by app/ml/train.py on our own labeled
            # data — still the same permitted ImageNet-pretrained backbone,
            # just with its last block adapted; no defect-specific
            # pretrained checkpoint is loaded anywhere.
            self.feature_extractor.load_state_dict(
                torch.load(BACKBONE_WEIGHTS_PATH, map_location="cpu", weights_only=True)
            )
        self.pool = nn.AdaptiveAvgPool2d(1)
        for p in self.feature_extractor.parameters():
            p.requires_grad = False
        self.feature_extractor.eval()

        self.head = DefectHead(in_features=1280)
        if WEIGHTS_PATH.exists():
            self.head.load_state_dict(torch.load(WEIGHTS_PATH, map_location="cpu", weights_only=True))
        else:
            print(f"[warn] no trained head found at {WEIGHTS_PATH} — run app/ml/train.py first. "
                  f"Predictions will be untrained/random until then.")
        self.head.eval()

    def _embed(self, img: Image.Image) -> torch.Tensor:
        x = _transform(img.convert("RGB")).unsqueeze(0)
        with torch.no_grad():
            feats = self.feature_extractor(x)
            feats = self.pool(feats).flatten(1)
        return feats

    def predict(self, image_path: str):
        img = Image.open(image_path)
        feats = self._embed(img)
        with torch.no_grad():
            logits = self.head(feats)
            probs = torch.softmax(logits, dim=1).squeeze(0)
        idx = int(torch.argmax(probs).item())
        return CLASSES[idx], float(probs[idx].item())


_classifier_singleton: DefectClassifier | None = None


def get_classifier() -> DefectClassifier:
    global _classifier_singleton
    if _classifier_singleton is None:
        _classifier_singleton = DefectClassifier()
    return _classifier_singleton
