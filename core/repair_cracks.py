from __future__ import annotations

import cv2
import numpy as np


def repair_cracks(image: np.ndarray) -> tuple[np.ndarray, dict[str, object]]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    binary = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        35,
        3,
    )
    closed = cv2.morphologyEx(
        binary,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        iterations=2,
    )
    closed = cv2.medianBlur(closed, 3)

    light = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(light)
    repaired_l = cv2.max(l_channel, closed)
    repaired_lab = cv2.merge((repaired_l, a_channel, b_channel))
    repaired = cv2.cvtColor(repaired_lab, cv2.COLOR_LAB2BGR)
    repaired = cv2.medianBlur(repaired, 3)

    crack_fill_ratio = float(np.count_nonzero(closed)) / float(closed.size)
    return repaired, {"crack_fill_ratio": round(crack_fill_ratio, 4)}
