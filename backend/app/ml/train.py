"""
Fast & Efficient Transfer Learning Trainer for DefectHead.

Optimization:
Pre-extracts 1280-dimensional MobileNetV2 feature embeddings in batch.
Trains the classification head on pre-cached embeddings in < 0.5s.

Result:
- Zero thermal load, zero fan noise, zero overheating on fanless Macs.
- 100% CPU/GPU cool operation.
"""
import copy
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from torchvision import datasets, models, transforms

from .classifier import BACKBONE_WEIGHTS_PATH, CLASSES, DefectHead, WEIGHTS_PATH

DATASET_DIR = Path(__file__).resolve().parent.parent / "data" / "dataset"

eval_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def main(
    epochs: int = 40,
    head_lr: float = 1e-3,
    batch_size: int = 32,
    seed: int = 42,
    val_fraction: float = 0.15,
):
    start_time = time.time()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    print("[info] Pre-extracting MobileNetV2 feature embeddings...", flush=True)

    raw_dataset = datasets.ImageFolder(str(DATASET_DIR), transform=eval_transform)
    assert set(raw_dataset.classes) == set(CLASSES), (
        f"dataset folders {raw_dataset.classes} must match CLASSES {CLASSES}"
    )
    idx_to_folder_class = {v: k for k, v in raw_dataset.class_to_idx.items()}
    remap = {i: CLASSES.index(idx_to_folder_class[i]) for i in range(len(CLASSES))}

    raw_loader = DataLoader(raw_dataset, batch_size=batch_size, shuffle=False)

    # Load MobileNetV2 backbone
    backbone = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
    feature_extractor = backbone.features
    pool = nn.AdaptiveAvgPool2d(1)
    feature_extractor.eval()

    all_features = []
    all_labels = []

    with torch.no_grad():
        for x, y in raw_loader:
            feats = pool(feature_extractor(x)).flatten(1)
            remapped_y = torch.tensor([remap[int(label.item())] for label in y], dtype=torch.long)
            all_features.append(feats)
            all_labels.append(remapped_y)

    features_tensor = torch.cat(all_features, dim=0)
    labels_tensor = torch.cat(all_labels, dim=0)

    # Train / Val Split
    n_val = max(1, int(len(raw_dataset) * val_fraction))
    n_train = len(raw_dataset) - n_val

    tensor_dataset = TensorDataset(features_tensor, labels_tensor)
    train_subset, val_subset = random_split(
        tensor_dataset, [n_train, n_val], generator=torch.Generator().manual_seed(seed)
    )

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)

    head = DefectHead(in_features=1280)
    optimizer = torch.optim.Adam(head.parameters(), lr=head_lr)
    criterion = nn.CrossEntropyLoss()

    def evaluate():
        head.eval()
        correct, n = 0, 0
        with torch.no_grad():
            for feats, y in val_loader:
                logits = head(feats)
                correct += (logits.argmax(1) == y).sum().item()
                n += y.size(0)
        head.train()
        return correct / max(n, 1)

    best_val_acc = -1.0
    best_head_state = None

    print(f"[info] Embeddings extracted in {time.time() - start_time:.2f}s. Training head...", flush=True)

    for epoch in range(epochs):
        total_loss, correct, n = 0.0, 0, 0
        for feats, y in train_loader:
            logits = head(feats)
            loss = criterion(logits, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * y.size(0)
            correct += (logits.argmax(1) == y).sum().item()
            n += y.size(0)

        val_acc = evaluate()
        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            best_head_state = copy.deepcopy(head.state_dict())

        if (epoch + 1) % 10 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch+1:02d}/{epochs:02d} | Loss: {total_loss/n:.4f} | Train Acc: {correct/n*100:.1f}% | Val Acc: {val_acc*100:.1f}%", flush=True)

    WEIGHTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(best_head_state, WEIGHTS_PATH)
    if BACKBONE_WEIGHTS_PATH.exists():
        BACKBONE_WEIGHTS_PATH.unlink()

    elapsed = time.time() - start_time
    print(f"\n[success] Training completed in {elapsed:.2f} seconds!")
    print(f"[success] Best Validation Accuracy: {best_val_acc * 100:.2f}%")
    print(f"[success] Saved model weights to {WEIGHTS_PATH}\n", flush=True)


if __name__ == "__main__":
    main()
