from __future__ import annotations

import cv2
import numpy as np


def _remove_small_components(mask: np.ndarray, minimum_area: int) -> np.ndarray:
    components, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    cleaned = np.zeros_like(mask)

    for component_index in range(1, components):
        area = stats[component_index, cv2.CC_STAT_AREA]
        if area >= minimum_area:
            cleaned[labels == component_index] = 255

    return cleaned


def cleanup_image(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    lightness, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    enhanced_lightness = clahe.apply(lightness)
    enhanced = cv2.cvtColor(cv2.merge((enhanced_lightness, a_channel, b_channel)), cv2.COLOR_LAB2BGR)

    gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
    adaptive = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        41,
        5,
    )
    _, otsu = cv2.threshold(
        cv2.GaussianBlur(gray, (5, 5), 0),
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
    )

    mask = cv2.bitwise_or(adaptive, otsu)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    minimum_area = max(int(image.shape[0] * image.shape[1] * 0.0004), 64)
    mask = _remove_small_components(mask, minimum_area)
    mask = cv2.GaussianBlur(mask, (5, 5), 0)
    _, mask = cv2.threshold(mask, 80, 255, cv2.THRESH_BINARY)

    prepared = cv2.GaussianBlur(enhanced, (3, 3), 0)
    preview = prepared.copy()
    preview[mask == 0] = 255

    meta = {
        "maskCoverage": round(float(np.count_nonzero(mask)) / float(mask.size), 4),
        "minimumArea": int(minimum_area),
    }
    return preview, mask, meta

