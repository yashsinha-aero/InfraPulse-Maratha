"""
Classical-CV severity/extent scoring (0-100). Deterministic and defect-type
aware — no learned model, so it's fully explainable in the documentation.

- spalling / cracked_tiles: edge/crack density via Canny edge detection.
- stagnant_water / paint_peeling: discolored/wet surface area via HSV
  color-range segmentation.
"""
import cv2
import numpy as np


def _edge_density_score(img_bgr: np.ndarray) -> float:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, 50, 150)
    density = edges.mean() / 255.0  # fraction of edge pixels
    return float(np.clip(density * 400, 0, 100))  # scaled empirically


def _color_area_score(img_bgr: np.ndarray, defect_label: str) -> float:
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    h, w = img_bgr.shape[:2]
    total = h * w

    if defect_label == "stagnant_water":
        # water: low saturation dark/reflective patches, or blue-ish tint
        lower1 = np.array([0, 0, 20])
        upper1 = np.array([180, 60, 160])
        mask = cv2.inRange(hsv, lower1, upper1)
    else:  # paint_peeling: patchy discoloration / exposed surface contrast
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    area_fraction = float(np.count_nonzero(mask)) / total
    return float(np.clip(area_fraction * 150, 0, 100))


def compute_severity(image_path: str, defect_label: str) -> float:
    img = cv2.imread(image_path)
    if img is None:
        return 0.0

    if defect_label in ("spalling", "cracked_tiles"):
        return round(_edge_density_score(img), 2)
    return round(_color_area_score(img, defect_label), 2)
