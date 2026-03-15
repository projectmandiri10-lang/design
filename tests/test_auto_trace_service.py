from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from core.auto_trace_service import AutoTraceService
from core.types import PipelineConfig, PipelinePaths


def test_auto_trace_service_runs_with_fallbacks(tmp_path: Path) -> None:
    image = np.zeros((320, 320, 3), dtype=np.uint8)
    yy, xx = np.indices((320, 320))
    image[:, :, 0] = np.clip(50 + xx * 0.45, 0, 255).astype(np.uint8)
    image[:, :, 1] = np.clip(40 + yy * 0.55, 0, 255).astype(np.uint8)
    image[:, :, 2] = np.clip(25 + (xx + yy) * 0.25, 0, 255).astype(np.uint8)
    cv2.circle(image, (160, 160), 96, (20, 70, 210), thickness=-1)
    cv2.rectangle(image, (88, 82), (235, 248), (215, 230, 245), thickness=14)
    cv2.line(image, (30, 260), (290, 40), (0, 0, 0), thickness=5)

    image_path = tmp_path / "shirt.png"
    assert cv2.imwrite(str(image_path), image)

    config = PipelineConfig(
        paths=PipelinePaths(
            detector_model=tmp_path / "missing_detector.onnx",
            realesrgan_model=tmp_path / "missing_realesrgan.onnx",
            unet_model=tmp_path / "missing_unet.onnx",
            potrace_bin=None,
            inkscape_bin=None,
        ),
        color_count=2,
    )
    service = AutoTraceService(config)
    result = service.run(image_path)

    assert result.vector_svg.startswith("<svg")
    assert len(result.layers) >= 1
    assert "edge" in result.previews
    assert result.metadata.get("pipelineMode") == "photo"
