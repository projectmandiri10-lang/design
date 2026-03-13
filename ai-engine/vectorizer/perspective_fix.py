from __future__ import annotations

import cv2
import numpy as np


def _order_points(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32)
    sums = points.sum(axis=1)
    diffs = np.diff(points, axis=1).reshape(-1)

    ordered = np.zeros((4, 2), dtype=np.float32)
    ordered[0] = points[np.argmin(sums)]
    ordered[2] = points[np.argmax(sums)]
    ordered[1] = points[np.argmin(diffs)]
    ordered[3] = points[np.argmax(diffs)]
    return ordered


def _distance(first: np.ndarray, second: np.ndarray) -> float:
    return float(np.linalg.norm(first - second))


def correct_perspective(image):
    height, width = image.shape[:2]
    image_area = float(height * width)
    meta = {
        "applied": False,
        "confidence": 0.0,
    }

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    edges = cv2.dilate(edges, None, iterations=1)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return image, meta

    best_box = None
    best_confidence = 0.0

    for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:10]:
        contour_area = cv2.contourArea(contour)
        if contour_area < image_area * 0.03:
            continue

        perimeter = cv2.arcLength(contour, True)
        approximation = cv2.approxPolyDP(contour, 0.02 * perimeter, True)

        if len(approximation) == 4:
            box = approximation.reshape(4, 2).astype(np.float32)
        else:
            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect).astype(np.float32)

        ordered = _order_points(box)
        top_width = _distance(ordered[0], ordered[1])
        bottom_width = _distance(ordered[3], ordered[2])
        left_height = _distance(ordered[0], ordered[3])
        right_height = _distance(ordered[1], ordered[2])

        warp_width = int(max(top_width, bottom_width))
        warp_height = int(max(left_height, right_height))

        if warp_width < 64 or warp_height < 64:
            continue

        aspect_ratio = warp_width / max(warp_height, 1)
        if aspect_ratio < 0.15 or aspect_ratio > 6.0:
            continue

        confidence = contour_area / image_area
        if confidence > best_confidence:
            best_confidence = confidence
            best_box = ordered

    if best_box is None or best_confidence < 0.05:
        return image, meta

    max_width = int(max(_distance(best_box[0], best_box[1]), _distance(best_box[2], best_box[3])))
    max_height = int(max(_distance(best_box[0], best_box[3]), _distance(best_box[1], best_box[2])))

    destination = np.array(
        [
            [0, 0],
            [max_width - 1, 0],
            [max_width - 1, max_height - 1],
            [0, max_height - 1],
        ],
        dtype=np.float32,
    )

    matrix = cv2.getPerspectiveTransform(best_box, destination)
    warped = cv2.warpPerspective(image, matrix, (max_width, max_height))

    meta["applied"] = True
    meta["confidence"] = round(best_confidence, 4)
    meta["outputShape"] = [int(warped.shape[1]), int(warped.shape[0])]
    return warped, meta

