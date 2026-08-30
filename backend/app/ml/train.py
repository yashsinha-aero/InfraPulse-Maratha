"""
Trains the DefectHead on top of the frozen MobileNetV2 backbone using images
placed in app/data/dataset/<class_name>/*.jpg. Holds out a small validation
split and keeps the best-val-accuracy checkpoint.

The backbone stays ImageNet-pretrained and frozen by default. Passing
`finetune_last_n_blocks > 0` unfreezes that many trailing blocks and
fine-tunes them at a small learning rate — this was tried and did not
improve held-out accuracy on our small dataset (it overfits), so it is off
by default; the option is kept for future experimentation.

Usage:
    python -m app.ml.train
"""
import copy
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, models, transforms

from .classifier import BACKBONE_WEIGHTS_PATH, CLASSES, DefectHead, WEIGHTS_PATH

DATASET_DIR = Path(__file__).resolve().parent.parent / "data" / "dataset"

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def main(
    epochs: int = 30,
    head_lr: float = 1e-3,
    backbone_lr: float = 1e-5,
    batch_size: int = 16,
    seed: int = 42,
    val_fraction: float = 0.15,
    finetune_last_n_blocks: int = 0,
):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    full_dataset = datasets.ImageFolder(str(DATASET_DIR))
    assert set(full_dataset.classes) == set(CLASSES), (
        f"dataset folders {full_dataset.classes} must match CLASSES {CLASSES} "
        f"(one subfolder per class, e.g. app/data/dataset/spalling/*.jpg)"
    )
    idx_to_folder_class = {v: k for k, v in full_dataset.class_to_idx.items()}
    remap = {i: CLASSES.index(idx_to_folder_class[i]) for i in range(len(CLASSES))}

    n_val = max(1, int(len(full_dataset) * val_fraction))
    n_train = len(full_dataset) - n_val
    train_subset, val_subset = random_split(
        full_dataset, [n_train, n_val], generator=torch.Generator().manual_seed(seed)
    )

    class Wrapped(torch.utils.data.Dataset):
        def __init__(self, subset, transform):
            self.subset = subset
            self.transform = transform

        def __len__(self):
            return len(self.subset)

        def __getitem__(self, i):
            img, label = self.subset[i]
            return self.transform(img.convert("RGB")), remap[label]

    train_loader = DataLoader(Wrapped(train_subset, train_transform), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(Wrapped(val_subset, val_transform), batch_size=batch_size, shuffle=False)

    backbone = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
    feature_extractor = backbone.features
    pool = nn.AdaptiveAvgPool2d(1)
    for p in feature_extractor.parameters():
        p.requires_grad = False
    if finetune_last_n_blocks > 0:
        for block in feature_extractor[-finetune_last_n_blocks:]:
            for p in block.parameters():
                p.requires_grad = True
    # eval() keeps BatchNorm running stats frozen (safe with tiny batches);
    # gradients still flow through any unfrozen block's weights/affine params.
    feature_extractor.eval()

    head = DefectHead(in_features=1280)
    param_groups = [{"params": head.parameters(), "lr": head_lr}]
    trainable_backbone_params = [p for p in feature_extractor.parameters() if p.requires_grad]
    if trainable_backbone_params:
        param_groups.append({"params": trainable_backbone_params, "lr": backbone_lr})
    optimizer = torch.optim.Adam(param_groups)
    criterion = nn.CrossEntropyLoss()

    def embed(x, grad_enabled):
        if grad_enabled:
            feats = pool(feature_extractor(x)).flatten(1)
        else:
            with torch.no_grad():
                feats = pool(feature_extractor(x)).flatten(1)
        return feats

    def evaluate():
        head.eval()
        correct, n = 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                logits = head(embed(x, grad_enabled=False))
                correct += (logits.argmax(1) == y).sum().item()
                n += y.size(0)
        head.train()
        return correct / max(n, 1)

    best_val_acc = -1.0
    best_head_state = None
    best_backbone_state = None

    for epoch in range(epochs):
        total_loss, correct, n = 0.0, 0, 0
        for x, y in train_loader:
            feats = embed(x, grad_enabled=True)
            logits = head(feats)
            loss = criterion(logits, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * x.size(0)
            correct += (logits.argmax(1) == y).sum().item()
            n += x.size(0)

        val_acc = evaluate()
        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            best_head_state = copy.deepcopy(head.state_dict())
            if finetune_last_n_blocks > 0:
                best_backbone_state = copy.deepcopy(feature_extractor.state_dict())

        print(f"epoch {epoch+1}/{epochs}  loss={total_loss/n:.4f}  "
              f"train_acc={correct/n:.4f}  val_acc={val_acc:.4f}")

    WEIGHTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(best_head_state, WEIGHTS_PATH)
    print(f"best val_acc={best_val_acc:.4f}")
    print(f"saved trained head to {WEIGHTS_PATH}")
    if finetune_last_n_blocks > 0:
        torch.save(best_backbone_state, BACKBONE_WEIGHTS_PATH)
        print(f"saved fine-tuned backbone to {BACKBONE_WEIGHTS_PATH}")
    elif BACKBONE_WEIGHTS_PATH.exists():
        BACKBONE_WEIGHTS_PATH.unlink()  # stale fine-tuned backbone from a prior run


if __name__ == "__main__":
    main()
