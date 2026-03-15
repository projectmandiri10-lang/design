from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from config import build_default_pipeline_config
from core.detect_sablon_area import SablonAreaDetector
from modules.ai_cleanup import CleanupArtifacts
from modules.color_reduce import reduce_palette
from modules.edge_detect import detect_outline
from modules.export_svg import export_timestamped_svg
from modules.preprocess import prepare_photo_input
from modules.vectorize import generate_vector_svg


def _photo_image() -> np.ndarray:
    image = np.full((420, 340, 3), (36, 34, 34), dtype=np.uint8)
    cv2.rectangle(image, (72, 88), (270, 332), (232, 232, 232), thickness=-1)
    cv2.circle(image, (168, 168), 62, (28, 108, 216), thickness=-1)
    cv2.putText(image, "AI", (132, 265), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (28, 28, 28), 5, cv2.LINE_AA)
    cv2.line(image, (90, 122), (254, 304), (86, 198, 94), 8, cv2.LINE_AA)
    return image


def test_prompt_style_modules_pipeline_produces_stage_artifacts(tmp_path: Path) -> None:
    image_path = tmp_path / "shirt.png"
    assert cv2.imwrite(str(image_path), _photo_image())

    config = build_default_pipeline_config(tmp_path)
    config.paths.detector_model = tmp_path / "missing_detector.onnx"
    config.paths.realesrgan_model = tmp_path / "missing_realesrgan.onnx"
    config.paths.unet_model = tmp_path / "missing_unet.onnx"
    config.controlnet.enabled = False

    detector = SablonAreaDetector(config.paths.detector_model)
    preprocess = prepare_photo_input(image_path, config, detector)
    assert preprocess.perspective_image.size > 0
    assert preprocess.original_preview.shape[0] > 0

    cleanup = CleanupArtifacts(
        cleaned_image=preprocess.perspective_image.copy(),
        metadata={"backend": "test", "usedFallback": True},
    )
    outline = detect_outline(cleanup.cleaned_image, config)
    assert outline.edge_map.shape[:2] == cleanup.cleaned_image.shape[:2]
    assert outline.foreground_mask.shape[:2] == cleanup.cleaned_image.shape[:2]

    colors = reduce_palette(
        cleanup.cleaned_image,
        foreground_mask=outline.foreground_mask,
        color_count=6,
        config=config,
    )
    assert colors.quantized.shape == cleanup.cleaned_image.shape
    assert len(colors.palette) >= 1

    vector = generate_vector_svg(
        label_map=colors.label_map,
        palette=colors.palette,
        canvas_size=(colors.quantized.shape[1], colors.quantized.shape[0]),
        config=config,
        title="AUTO VECTOR SABLON AI Test",
    )
    assert vector.vector_svg.startswith("<svg")
    assert len(vector.layers) >= 1

    saved = export_timestamped_svg(vector.vector_svg, tmp_path / "output")
    assert saved.exists()
    assert saved.name.startswith("vector_")
    assert saved.suffix == ".svg"
