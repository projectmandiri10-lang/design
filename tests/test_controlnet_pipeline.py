from __future__ import annotations

from pathlib import Path
from types import MethodType

import cv2
import numpy as np

from core.auto_trace_service import AutoTraceService
from core.controlnet_client import ControlNetClientError
from core.types import PipelineConfig, PipelinePaths


def _config(tmp_path: Path) -> PipelineConfig:
    return PipelineConfig(
        paths=PipelinePaths(
            detector_model=tmp_path / "missing_detector.onnx",
            realesrgan_model=tmp_path / "missing_realesrgan.onnx",
            unet_model=tmp_path / "missing_unet.onnx",
            potrace_bin=None,
            inkscape_bin=None,
        ),
        color_count=2,
    )


def _photo_image() -> np.ndarray:
    image = np.full((360, 320, 3), (35, 35, 35), dtype=np.uint8)
    cv2.rectangle(image, (76, 74), (244, 294), (235, 235, 235), thickness=-1)
    cv2.circle(image, (160, 154), 54, (28, 108, 215), thickness=-1)
    cv2.putText(image, "DTF", (106, 256), cv2.FONT_HERSHEY_SIMPLEX, 1.55, (25, 25, 25), 4, cv2.LINE_AA)
    cv2.line(image, (92, 110), (232, 278), (88, 202, 92), 8, cv2.LINE_AA)
    return image


def test_photo_pipeline_uses_controlnet_outline_and_keeps_repaired_for_colors(tmp_path: Path) -> None:
    image_path = tmp_path / "shirt.png"
    assert cv2.imwrite(str(image_path), _photo_image())

    config = _config(tmp_path)
    config.controlnet.enabled = True
    config.controlnet.preprocessor = "lineart"

    service = AutoTraceService(config)
    captured: dict[str, object] = {}
    outline_image = np.full((64, 64, 3), 255, dtype=np.uint8)

    def fake_controlnet(self, image: np.ndarray) -> tuple[np.ndarray, dict[str, object]]:
        nonlocal outline_image
        outline_image = np.full_like(image, 255)
        cv2.rectangle(outline_image, (24, 24), (image.shape[1] - 24, image.shape[0] - 24), (0, 0, 0), thickness=10)
        return outline_image, {
            "requested": True,
            "applied": True,
            "preprocessor": "lineart",
            "model": "control_v11p_sd15_lineart [hash]",
            "baseUrl": "http://127.0.0.1:7860",
            "fallbackReason": None,
        }

    original_get_color_reduction = service._get_color_reduction

    def wrapped_get_color_reduction(self, image, foreground_mask, image_state, preset_used):
        captured["used_controlnet_image"] = np.array_equal(image, outline_image)
        return original_get_color_reduction(image, foreground_mask, image_state, preset_used)

    service._run_controlnet_outline = MethodType(fake_controlnet, service)
    service._get_color_reduction = MethodType(wrapped_get_color_reduction, service)

    result = service.run(image_path)

    assert result.metadata["pipelineMode"] == "photo"
    assert result.metadata["outlineBackend"] == "controlnet"
    assert result.metadata["controlnet"]["applied"] is True
    assert result.metadata["controlnet"]["preprocessor"] == "lineart"
    assert "controlnet_outline" in result.previews
    assert captured["used_controlnet_image"] is False


def test_controlnet_fallback_warns_and_uses_classic_outline(tmp_path: Path) -> None:
    image_path = tmp_path / "shirt.png"
    assert cv2.imwrite(str(image_path), _photo_image())

    config = _config(tmp_path)
    config.controlnet.enabled = True
    service = AutoTraceService(config)

    def failing_controlnet(self, image: np.ndarray) -> tuple[np.ndarray, dict[str, object]]:
        del image
        raise ControlNetClientError("server unavailable")

    service._run_controlnet_outline = MethodType(failing_controlnet, service)
    result = service.run(image_path)

    assert result.metadata["outlineBackend"] == "classic"
    assert result.metadata["controlnet"]["applied"] is False
    assert result.metadata["controlnet"]["fallbackReason"] == "server unavailable"
    assert "controlnet_outline" not in result.previews
    assert any("ControlNet fallback" in warning for warning in result.warnings)


def test_controlnet_is_ignored_for_transparent_artwork(tmp_path: Path) -> None:
    rgba = np.zeros((240, 240, 4), dtype=np.uint8)
    cv2.rectangle(rgba, (26, 34), (214, 214), (255, 255, 255, 255), thickness=-1)
    cv2.circle(rgba, (120, 118), 58, (0, 190, 255, 255), thickness=-1)
    image_path = tmp_path / "design.png"
    assert cv2.imwrite(str(image_path), rgba)

    config = _config(tmp_path)
    config.controlnet.enabled = True
    service = AutoTraceService(config)
    called = {"value": False}

    def fake_controlnet(self, image: np.ndarray) -> tuple[np.ndarray, dict[str, object]]:
        del image
        called["value"] = True
        raise AssertionError("ControlNet should not run for transparent artwork")

    service._run_controlnet_outline = MethodType(fake_controlnet, service)
    result = service.run(image_path)

    assert result.metadata["pipelineMode"] == "artwork"
    assert result.metadata["outlineBackend"] == "classic"
    assert result.metadata["controlnet"]["applied"] is False
    assert "photo pipeline" in str(result.metadata["controlnet"]["fallbackReason"]).lower()
    assert called["value"] is False


def test_controlnet_is_ignored_for_raster_artwork(tmp_path: Path) -> None:
    image = np.full((320, 260, 3), 255, dtype=np.uint8)
    cv2.circle(image, (145, 160), 78, (24, 24, 24), thickness=6)
    cv2.putText(image, "LOGO", (48, 184), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (30, 30, 30), 4, cv2.LINE_AA)
    image_path = tmp_path / "logo.jpg"
    assert cv2.imwrite(str(image_path), image, [int(cv2.IMWRITE_JPEG_QUALITY), 92])

    config = _config(tmp_path)
    config.controlnet.enabled = True
    service = AutoTraceService(config)
    called = {"value": False}

    def fake_controlnet(self, image: np.ndarray) -> tuple[np.ndarray, dict[str, object]]:
        del image
        called["value"] = True
        raise AssertionError("ControlNet should not run for raster artwork")

    service._run_controlnet_outline = MethodType(fake_controlnet, service)
    result = service.run(image_path)

    assert result.metadata["pipelineMode"] == "raster_artwork"
    assert result.metadata["outlineBackend"] == "classic"
    assert result.metadata["controlnet"]["applied"] is False
    assert "photo pipeline" in str(result.metadata["controlnet"]["fallbackReason"]).lower()
    assert called["value"] is False
