from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from core.detect_sablon_area import SablonAreaDetector
from core.image_preprocess import compose_alpha_preview, load_image_data, resize_loaded_image
from core.perspective_correction import correct_perspective
from core.types import DetectionResult, LoadedImage, PipelineConfig


@dataclass(slots=True)
class PreprocessArtifacts:
    input_path: Path
    loaded: LoadedImage
    resized: LoadedImage
    resize_scale: float
    detection: DetectionResult
    perspective_image: np.ndarray
    original_preview: np.ndarray
    metadata: dict[str, object]


def prepare_photo_input(
    input_path: str | Path,
    config: PipelineConfig,
    detector: SablonAreaDetector,
) -> PreprocessArtifacts:
    image_path = Path(input_path)
    loaded = load_image_data(image_path)
    resized, resize_scale = resize_loaded_image(loaded, config.max_side_px)
    detection = detector.detect(resized.bgr)
    perspective_image, perspective_meta = correct_perspective(detection.cropped_image)
    original_preview = compose_alpha_preview(resized.bgr, resized.alpha_mask)
    metadata = {
        "resizeScale": float(resize_scale),
        "detection": {
            "bbox_xywh": tuple(int(value) for value in detection.bbox_xywh),
            "confidence": float(detection.confidence),
            "method": str(detection.method),
        },
        "perspective": perspective_meta,
    }
    return PreprocessArtifacts(
        input_path=image_path,
        loaded=loaded,
        resized=resized,
        resize_scale=resize_scale,
        detection=detection,
        perspective_image=perspective_image,
        original_preview=original_preview,
        metadata=metadata,
    )
