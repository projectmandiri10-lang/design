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


def _distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def correct_perspective(image: np.ndarray) -> tuple[np.ndarray, dict[str, object]]:
    height, width = image.shape[:2]
    image_area = float(height * width)
    meta: dict[str, object] = {"applied": False, "confidence": 0.0}

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edge = cv2.Canny(blur, 60, 180)
    edge = cv2.morphologyEx(
        edge,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)),
        iterations=1,
    )

    contours, _ = cv2.findContours(edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return image, meta

    best_box = None
    best_score = 0.0

    for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:12]:
        area = cv2.contourArea(contour)
        if area < image_area * 0.03:
            continue

        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
        if len(approx) == 4:
            box = approx.reshape(4, 2).astype(np.float32)
        else:
            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect).astype(np.float32)

        ordered = _order_points(box)
        top_w = _distance(ordered[0], ordered[1])
        bottom_w = _distance(ordered[3], ordered[2])
        left_h = _distance(ordered[0], ordered[3])
        right_h = _distance(ordered[1], ordered[2])

        warp_w = int(max(top_w, bottom_w))
        warp_h = int(max(left_h, right_h))
        if warp_w < 64 or warp_h < 64:
            continue

        aspect = warp_w / max(warp_h, 1)
        if aspect < 0.15 or aspect > 6.5:
            continue

        score = float(area / image_area)
        if score > best_score:
            best_score = score
            best_box = ordered

    if best_box is None or best_score < 0.05:
        return image, meta

    max_w = int(max(_distance(best_box[0], best_box[1]), _distance(best_box[2], best_box[3])))
    max_h = int(max(_distance(best_box[0], best_box[3]), _distance(best_box[1], best_box[2])))

    destination = np.array(
        [
            [0, 0],
            [max_w - 1, 0],
            [max_w - 1, max_h - 1],
            [0, max_h - 1],
        ],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(best_box, destination)
    warped = cv2.warpPerspective(image, matrix, (max_w, max_h), flags=cv2.INTER_CUBIC)
    meta["applied"] = True
    meta["confidence"] = round(best_score, 4)
    meta["output_shape"] = [int(warped.shape[1]), int(warped.shape[0])]
    return warped, meta
