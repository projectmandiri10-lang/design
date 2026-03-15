from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from core.detect_sablon_area import SablonAreaDetector


def test_contour_detector_fallback() -> None:
    image = np.full((240, 240, 3), 255, dtype=np.uint8)
    cv2.rectangle(image, (50, 70), (190, 180), (10, 10, 10), thickness=-1)
    cv2.putText(image, "LOGO", (70, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)

    detector = SablonAreaDetector(Path("missing_detector.onnx"))
    result = detector.detect(image)

    assert result.cropped_image.size > 0
    assert result.bbox_xywh[2] > 80
    assert result.bbox_xywh[3] > 60
