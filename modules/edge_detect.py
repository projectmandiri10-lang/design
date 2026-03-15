from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from core.controlnet_client import ControlNetClient, ControlNetClientError
from core.image_preprocess import preprocess_ai_outline, preprocess_for_outline
from core.types import PipelineConfig


@dataclass(slots=True)
class OutlineArtifacts:
    preview_image: np.ndarray
    edge_map: np.ndarray
    foreground_mask: np.ndarray
    metadata: dict[str, object]


def detect_outline(image: np.ndarray, config: PipelineConfig) -> OutlineArtifacts:
    fallback_reason: str | None = None

    if config.controlnet.enabled:
        try:
            rendered = ControlNetClient(config.controlnet).render_outline(image)
            preprocess = preprocess_ai_outline(rendered.image, config, preset_used="photo")
            return OutlineArtifacts(
                preview_image=rendered.image,
                edge_map=preprocess.edge_map,
                foreground_mask=preprocess.foreground_mask,
                metadata={
                    "backend": "controlnet",
                    "preprocessor": rendered.discovery.module,
                    "model": rendered.discovery.model,
                    "baseUrl": rendered.discovery.base_url,
                    "fallbackReason": None,
                },
            )
        except ControlNetClientError as error:
            fallback_reason = str(error)

    preprocess = preprocess_for_outline(image, config, preset_used="photo")
    preview_image = preprocess.edge_map
    if preview_image.ndim == 2:
        preview_image = cv2.cvtColor(preview_image, cv2.COLOR_GRAY2BGR)
    return OutlineArtifacts(
        preview_image=preview_image,
        edge_map=preprocess.edge_map,
        foreground_mask=preprocess.foreground_mask,
        metadata={
            "backend": "opencv_canny",
            "preprocessor": "canny",
            "model": None,
            "baseUrl": config.controlnet.normalized_base_url(),
            "fallbackReason": fallback_reason,
        },
    )
