"""
Evaluates the DefectClassifier model on the dataset images in app/data/dataset.
Prints overall accuracy, per-class accuracy breakdown, and confidence scores.

Usage:
    cd backend
    .venv/bin/python -m app.ml.evaluate
"""
import os
from pathlib import Path
from app.ml.classifier import get_classifier, CLASSES

def evaluate():
    clf = get_classifier()
    dataset_dir = Path(__file__).resolve().parent.parent / "data" / "dataset"

    correct = 0
    total = 0
    per_class = {c: {"correct": 0, "total": 0} for c in CLASSES}
    misclassified = []

    print("==================================================")
    print("      INFRAPULSE MODEL ACCURACY EVALUATION        ")
    print("==================================================")

    for cls in CLASSES:
        cls_dir = dataset_dir / cls
        if not cls_dir.exists():
            print(f"[warn] Directory not found for class '{cls}': {cls_dir}")
            continue

        for img_name in sorted(os.listdir(cls_dir)):
            if img_name.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
                img_path = str(cls_dir / img_name)
                pred, conf = clf.predict(img_path)
                total += 1
                per_class[cls]["total"] += 1

                if pred == cls:
                    correct += 1
                    per_class[cls]["correct"] += 1
                else:
                    misclassified.append({
                        "file": img_name,
                        "true": cls,
                        "pred": pred,
                        "conf": conf
                    })

    overall_acc = (correct / total * 100) if total > 0 else 0
    print(f"\nOverall Test Accuracy: {correct}/{total} ({overall_acc:.2f}%)\n")
    print("Per-Class Accuracy Breakdown:")
    print("--------------------------------------------------")
    for cls, stats in per_class.items():
        acc = (stats["correct"] / stats["total"] * 100) if stats["total"] > 0 else 0
        print(f"  • {cls:<16}: {stats['correct']:>2}/{stats['total']:<2} ({acc:>6.2f}%)")

    if misclassified:
        print("\nMisclassified Sample Details:")
        print("--------------------------------------------------")
        for m in misclassified:
            print(f"  • {m['file']} (True: {m['true']} → Predicted: {m['pred']} @ {m['conf']*100:.1f}%)")

    print("==================================================\n")

if __name__ == "__main__":
    evaluate()
