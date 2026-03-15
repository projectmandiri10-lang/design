from __future__ import annotations

import cv2
import numpy as np

from .types import LayerResult, VectorPathData


def build_cutline_paths(
    layers: list[LayerResult],
    offset_px: int = 8,
    simplify_ratio: float = 0.004,
) -> tuple[list[VectorPathData], np.ndarray, dict[str, int | float]]:
    if not layers:
        raise ValueError("No layers available for cutline export.")

    mask_shape = layers[0].mask.shape[:2]
    combined = np.zeros(mask_shape, dtype=np.uint8)
    for layer in layers:
        if layer.mask.shape[:2] != mask_shape:
            raise ValueError("Layer masks must have identical dimensions for cutline export.")
        combined = cv2.bitwise_or(combined, np.where(layer.mask > 0, 255, 0).astype(np.uint8))

    combined = cv2.morphologyEx(
        combined,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        iterations=1,
    )
    if cv2.countNonZero(combined) == 0:
        raise ValueError("Printable foreground is empty. Cutline export aborted.")

    resolved_offset = max(0, int(offset_px))
    if resolved_offset > 0:
        kernel_size = max(3, resolved_offset * 2 + 1)
        combined = cv2.dilate(
            combined,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)),
            iterations=1,
        )

    contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)
    paths: list[VectorPathData] = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < max(48.0, float(resolved_offset + 1) * 18.0):
            continue
        epsilon = max(1.0, float(cv2.arcLength(contour, True)) * float(simplify_ratio))
        approx = cv2.approxPolyDP(contour, epsilon, True)
        if len(approx) < 3:
            continue
        points = approx.reshape(-1, 2)
        commands = [f"M {points[0][0]:.2f} {points[0][1]:.2f}"]
        for point in points[1:]:
            commands.append(f"L {point[0]:.2f} {point[1]:.2f}")
        commands.append("Z")
        paths.append(VectorPathData(d=" ".join(commands)))

    paths = _sort_paths_by_area(paths)
    metadata = {
        "offsetPx": int(resolved_offset),
        "pathCount": int(len(paths)),
        "foregroundPixels": int(cv2.countNonZero(combined)),
    }
    return paths, combined, metadata


def _sort_paths_by_area(paths: list[VectorPathData]) -> list[VectorPathData]:
    def area_key(item: VectorPathData) -> float:
        points = _extract_points(item.d)
        if len(points) < 3:
            return 0.0
        return abs(cv2.contourArea(np.asarray(points, dtype=np.float32)))

    return sorted(paths, key=area_key, reverse=True)


def _extract_points(path_data: str) -> list[tuple[float, float]]:
    tokens = path_data.replace("M", " M ").replace("L", " L ").replace("Z", " Z ").split()
    points: list[tuple[float, float]] = []
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if token in {"M", "L"} and index + 2 < len(tokens):
            try:
                x = float(tokens[index + 1])
                y = float(tokens[index + 2])
                points.append((x, y))
            except ValueError:
                pass
            index += 3
            continue
        index += 1
    return points
