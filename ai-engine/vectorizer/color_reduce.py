from __future__ import annotations

import cv2
import numpy as np

from .io_utils import bgr_to_hex


def _ensure_foreground_mask(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    if np.count_nonzero(mask) > 0:
        return mask

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    fallback = np.where(gray < 250, 255, 0).astype(np.uint8)
    return fallback


def reduce_colors(image, mask, color_count: int):
    mask = _ensure_foreground_mask(image, mask)
    pixels = image[mask > 0].reshape(-1, 3)
    if pixels.size == 0:
        raise ValueError("No printable foreground could be isolated from the image.")

    pixels_float = np.float32(pixels)
    unique_count = len(np.unique(pixels, axis=0))
    cluster_count = max(1, min(color_count, unique_count))

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 25, 0.4)
    _compactness, labels, centers = cv2.kmeans(
        pixels_float,
        cluster_count,
        None,
        criteria,
        3,
        cv2.KMEANS_PP_CENTERS,
    )

    labels = labels.reshape(-1)
    centers = np.uint8(centers)
    raw_counts = np.bincount(labels, minlength=cluster_count)
    luminance = centers @ np.array([0.114, 0.587, 0.299])
    order = sorted(range(cluster_count), key=lambda index: (-raw_counts[index], luminance[index]))

    remap_lookup = np.zeros(cluster_count, dtype=np.uint8)
    for remapped_index, original_index in enumerate(order):
        remap_lookup[original_index] = remapped_index

    ordered_centers = centers[order]
    remapped_labels = remap_lookup[labels]

    smoothed_labels = np.zeros(mask.shape, dtype=np.uint8)
    smoothed_labels[mask > 0] = remapped_labels + 1
    smoothed_labels = cv2.medianBlur(smoothed_labels, 3)
    smoothed_labels[mask == 0] = 0

    label_map = np.full(mask.shape, -1, dtype=np.int16)
    label_map[smoothed_labels > 0] = smoothed_labels[smoothed_labels > 0] - 1

    quantized = np.full_like(image, 255)
    foreground = label_map >= 0
    quantized[foreground] = ordered_centers[label_map[foreground]]

    final_counts = np.bincount(label_map[foreground], minlength=cluster_count) if np.any(foreground) else np.array([], dtype=np.int32)
    palette = [
        {
            "index": int(index),
            "hex": bgr_to_hex(color),
            "pixels": int(final_counts[index]) if index < len(final_counts) else 0,
        }
        for index, color in enumerate(ordered_centers)
    ]

    meta = {
        "requestedColorCount": int(color_count),
        "actualColorCount": int(cluster_count),
    }
    return quantized, label_map, palette, meta

