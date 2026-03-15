from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from core.auto_trace_service import AutoTraceService
from core.image_preprocess import load_image_data
from core.types import PipelineConfig, PipelinePaths


def test_load_image_data_detects_alpha(tmp_path: Path) -> None:
    rgba = np.zeros((120, 120, 4), dtype=np.uint8)
    cv2.circle(rgba, (60, 60), 35, (20, 180, 240, 255), thickness=-1)
    image_path = tmp_path / "alpha.png"
    assert cv2.imwrite(str(image_path), rgba)

    loaded = load_image_data(image_path)
    assert loaded.has_alpha is True
    assert loaded.alpha_mask is not None
    assert np.count_nonzero(loaded.alpha_mask) > 0
    assert loaded.transparent_ratio > 0.1


def test_auto_trace_uses_artwork_mode_for_transparent_png(tmp_path: Path) -> None:
    rgba = np.zeros((260, 260, 4), dtype=np.uint8)
    cv2.rectangle(rgba, (30, 40), (230, 230), (255, 255, 255, 255), thickness=-1)
    cv2.circle(rgba, (130, 130), 60, (0, 190, 255, 255), thickness=-1)
    cv2.putText(rgba, "A", (95, 160), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (40, 40, 40, 255), 5, cv2.LINE_AA)
    image_path = tmp_path / "transparent_design.png"
    assert cv2.imwrite(str(image_path), rgba)

    config = PipelineConfig(
        paths=PipelinePaths(
            detector_model=tmp_path / "missing_detector.onnx",
            realesrgan_model=tmp_path / "missing_realesrgan.onnx",
            unet_model=tmp_path / "missing_unet.onnx",
            potrace_bin=None,
            inkscape_bin=None,
        ),
        color_count=4,
    )
    service = AutoTraceService(config)
    result = service.run(image_path)

    assert result.metadata.get("pipelineMode") == "artwork"
    skipped = set(result.metadata.get("skippedStages", []))
    assert {"perspective_correction", "realesrgan_restore", "unet_texture_remove", "repair_cracks"}.issubset(skipped)
    assert len(result.layers) >= 2

    edge = result.previews["edge"]
    components, _ = cv2.connectedComponents((edge > 0).astype(np.uint8))
    assert components >= 4


def test_auto_trace_uses_raster_artwork_mode_for_white_background_illustration(tmp_path: Path) -> None:
    image = np.full((320, 260, 3), 255, dtype=np.uint8)
    cv2.circle(image, (145, 160), 78, (24, 24, 24), thickness=6)
    cv2.circle(image, (130, 145), 12, (24, 24, 24), thickness=4)
    cv2.circle(image, (165, 145), 12, (24, 24, 24), thickness=4)
    cv2.circle(image, (126, 148), 4, (24, 24, 24), thickness=-1)
    cv2.circle(image, (161, 148), 4, (24, 24, 24), thickness=-1)
    cv2.ellipse(image, (145, 188), (26, 18), 0, 10, 170, (24, 24, 24), thickness=4)
    cv2.line(image, (115, 225), (175, 242), (60, 60, 60), thickness=4)
    cv2.line(image, (120, 102), (170, 88), (120, 120, 120), thickness=3)
    cv2.circle(image, (60, 70), 22, (24, 24, 24), thickness=4)
    cv2.circle(image, (220, 78), 18, (24, 24, 24), thickness=4)
    image_path = tmp_path / "illustration.jpg"
    assert cv2.imwrite(str(image_path), image, [int(cv2.IMWRITE_JPEG_QUALITY), 92])

    config = PipelineConfig(
        paths=PipelinePaths(
            detector_model=tmp_path / "missing_detector.onnx",
            realesrgan_model=tmp_path / "missing_realesrgan.onnx",
            unet_model=tmp_path / "missing_unet.onnx",
            potrace_bin=None,
            inkscape_bin=None,
        ),
        color_count=4,
    )
    service = AutoTraceService(config)
    result = service.run(image_path)

    assert result.metadata.get("pipelineMode") == "raster_artwork"
    skipped = set(result.metadata.get("skippedStages", []))
    assert {"perspective_correction", "realesrgan_restore", "unet_texture_remove", "repair_cracks"}.issubset(skipped)
    assert result.metadata["detection"]["method"] == "raster_foreground_bbox"
    assert result.metadata["smallImageBoost"]["applied"] is True
    assert len(result.layers) >= 2
    assert sum(layer.node_count for layer in result.layers) >= 24

    edge = result.previews["edge"]
    components, _ = cv2.connectedComponents((edge > 0).astype(np.uint8))
    assert components >= 5


def test_auto_trace_uses_raster_artwork_mode_for_small_colored_logo(tmp_path: Path) -> None:
    image = np.full((486, 494, 3), 255, dtype=np.uint8)
    cv2.ellipse(image, (135, 190), (92, 116), 0, 22, 308, (120, 203, 255), thickness=16)
    cv2.ellipse(image, (238, 172), (122, 124), 0, 205, 350, (24, 42, 82), thickness=16)
    cv2.ellipse(image, (220, 245), (196, 92), 6, 168, 342, (132, 195, 74), thickness=14)
    cv2.rectangle(image, (132, 248), (325, 338), (110, 192, 72), thickness=14)
    cv2.line(image, (152, 258), (300, 258), (255, 255, 255), thickness=10)
    cv2.line(image, (160, 294), (292, 294), (255, 255, 255), thickness=10)
    cv2.line(image, (176, 248), (176, 338), (255, 255, 255), thickness=10)
    cv2.line(image, (224, 248), (224, 338), (255, 255, 255), thickness=10)
    cv2.line(image, (270, 248), (270, 338), (255, 255, 255), thickness=10)
    cv2.line(image, (120, 250), (96, 208), (28, 61, 112), thickness=12)
    cv2.line(image, (332, 248), (360, 154), (28, 61, 112), thickness=12)
    cv2.circle(image, (162, 370), 20, (44, 122, 188), thickness=8)
    cv2.circle(image, (276, 370), 20, (44, 122, 188), thickness=8)
    pts = np.array([[292, 336], [390, 402], [338, 412], [356, 466], [262, 392]], dtype=np.int32)
    cv2.polylines(image, [pts], True, (24, 42, 82), thickness=12)
    image_path = tmp_path / "logo.png"
    assert cv2.imwrite(str(image_path), image)

    config = PipelineConfig(
        paths=PipelinePaths(
            detector_model=tmp_path / "missing_detector.onnx",
            realesrgan_model=tmp_path / "missing_realesrgan.onnx",
            unet_model=tmp_path / "missing_unet.onnx",
            potrace_bin=None,
            inkscape_bin=None,
        ),
        color_count=4,
    )
    service = AutoTraceService(config)
    result = service.run(image_path)

    assert result.metadata.get("pipelineMode") == "raster_artwork"
    assert result.metadata["smallImageBoost"]["applied"] is True
    assert result.metadata["rasterArtwork"]["coarseColorBins"] <= 96
    assert len(result.layers) >= 3
    assert sum(layer.node_count for layer in result.layers) >= 60
