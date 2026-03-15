from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from core.image_preprocess import load_image_data
from core.trace_analysis import analyze_trace_input, resolve_preset, select_pipeline_mode
from core.types import PipelineConfig, PipelinePaths


def _config(tmp_path: Path) -> PipelineConfig:
    return PipelineConfig(
        paths=PipelinePaths(
            detector_model=tmp_path / "missing_detector.onnx",
            realesrgan_model=tmp_path / "missing_realesrgan.onnx",
            unet_model=tmp_path / "missing_unet.onnx",
            potrace_bin=None,
            inkscape_bin=None,
        )
    )


def test_trace_analysis_prefers_logo_for_small_white_background_logo(tmp_path: Path) -> None:
    image = np.full((320, 320, 3), 255, dtype=np.uint8)
    cv2.circle(image, (160, 160), 88, (32, 48, 110), thickness=12)
    cv2.ellipse(image, (160, 160), (115, 58), 0, 190, 355, (96, 180, 78), thickness=10)
    cv2.rectangle(image, (110, 132), (210, 212), (96, 180, 78), thickness=10)
    path = tmp_path / "logo.png"
    assert cv2.imwrite(str(path), image)

    loaded = load_image_data(path)
    analysis, raster = analyze_trace_input(loaded, _config(tmp_path))

    assert analysis.recommended_preset == "logo"
    assert resolve_preset("auto", analysis) == "logo"
    assert select_pipeline_mode("logo", loaded, analysis) == "raster_artwork"
    assert raster.is_candidate is True


def test_trace_analysis_prefers_photo_for_noisy_image(tmp_path: Path) -> None:
    base = np.full((420, 420, 3), 120, dtype=np.uint8)
    gradient = np.linspace(0, 100, 420, dtype=np.uint8)
    base[:, :, 0] = gradient[None, :]
    base[:, :, 1] = gradient[:, None]
    base[:, :, 2] = 200
    noise = np.random.default_rng(7).normal(0, 28, base.shape).astype(np.int16)
    image = np.clip(base.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    cv2.circle(image, (210, 210), 120, (15, 40, 180), thickness=-1)
    path = tmp_path / "photo.jpg"
    assert cv2.imwrite(str(path), image, [int(cv2.IMWRITE_JPEG_QUALITY), 92])

    loaded = load_image_data(path)
    analysis, _ = analyze_trace_input(loaded, _config(tmp_path))

    assert analysis.recommended_preset == "photo"
    assert select_pipeline_mode("photo", loaded, analysis) == "photo"


def test_trace_analysis_prefers_text_typo_for_white_background_text(tmp_path: Path) -> None:
    image = np.full((280, 520, 3), 255, dtype=np.uint8)
    cv2.putText(image, "SALE", (36, 175), cv2.FONT_HERSHEY_SIMPLEX, 3.4, (18, 18, 18), 10, cv2.LINE_AA)
    cv2.putText(image, "HOT", (70, 248), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (35, 35, 35), 5, cv2.LINE_AA)
    path = tmp_path / "text.png"
    assert cv2.imwrite(str(path), image)

    loaded = load_image_data(path)
    analysis, raster = analyze_trace_input(loaded, _config(tmp_path))

    assert analysis.recommended_preset == "text_typo"
    assert resolve_preset("auto", analysis) == "text_typo"
    assert select_pipeline_mode("text_typo", loaded, analysis) == "raster_artwork"
    assert raster.is_candidate is True


def test_trace_analysis_prefers_dtf_screen_print_for_shirt_print_like_image(tmp_path: Path) -> None:
    image = np.full((420, 320, 3), (34, 30, 30), dtype=np.uint8)
    cv2.rectangle(image, (70, 80), (250, 300), (230, 230, 230), thickness=-1)
    cv2.circle(image, (160, 170), 58, (30, 110, 220), thickness=-1)
    cv2.putText(image, "DTF", (112, 260), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (25, 25, 25), 4, cv2.LINE_AA)
    cv2.line(image, (80, 110), (240, 290), (80, 200, 90), 8, cv2.LINE_AA)
    path = tmp_path / "shirt_print.jpg"
    assert cv2.imwrite(str(path), image, [int(cv2.IMWRITE_JPEG_QUALITY), 94])

    loaded = load_image_data(path)
    analysis, _ = analyze_trace_input(loaded, _config(tmp_path))

    assert analysis.recommended_preset == "dtf_screen_print"


def test_trace_analysis_prefers_sticker_cutting_for_simple_shape_artwork(tmp_path: Path) -> None:
    rgba = np.zeros((320, 320, 4), dtype=np.uint8)
    star = np.array(
        [
            [160, 30],
            [198, 122],
            [295, 122],
            [215, 182],
            [246, 284],
            [160, 222],
            [74, 284],
            [105, 182],
            [25, 122],
            [122, 122],
        ],
        dtype=np.int32,
    )
    cv2.fillPoly(rgba, [star], (25, 25, 25, 255))
    cv2.polylines(rgba, [star], True, (255, 255, 255, 255), thickness=10, lineType=cv2.LINE_AA)
    path = tmp_path / "sticker.png"
    assert cv2.imwrite(str(path), rgba)

    loaded = load_image_data(path)
    analysis, _ = analyze_trace_input(loaded, _config(tmp_path))

    assert analysis.recommended_preset == "sticker_cutting"
    assert select_pipeline_mode("sticker_cutting", loaded, analysis) == "artwork"
