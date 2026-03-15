from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

PySide6_QtWidgets = pytest.importorskip("PySide6.QtWidgets")
QApplication = PySide6_QtWidgets.QApplication

from core.types import PipelineConfig, PipelinePaths
from ui.control_panel import TraceControlPanel


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


def _app() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def test_pipeline_config_signature_changes_for_controlnet_settings(tmp_path: Path) -> None:
    config = _config(tmp_path)
    base_signature = config.settings_signature()

    config.controlnet.enabled = True
    assert config.settings_signature() != base_signature

    enabled_signature = config.settings_signature()
    config.controlnet.preprocessor = "canny"
    assert config.settings_signature() != enabled_signature

    preprocessor_signature = config.settings_signature()
    config.controlnet.base_url = "http://localhost:7860/"
    assert config.settings_signature() != preprocessor_signature


def test_control_panel_applies_controlnet_settings_and_summary(tmp_path: Path) -> None:
    _app()
    config = _config(tmp_path)
    panel = TraceControlPanel()
    panel.sync_from_config(config)

    panel.controlnet_enable_check.setChecked(True)
    panel.controlnet_url_edit.setText("http://localhost:7860/")
    panel.controlnet_preprocessor_combo.setCurrentIndex(1)
    panel.apply_to_config(config)

    assert config.controlnet.enabled is True
    assert config.controlnet.normalized_base_url() == "http://localhost:7860"
    assert config.controlnet.preprocessor == "canny"
    assert panel.controlnet_model_label.text() == "control_v11p_sd15_canny"

    panel.update_result_summary(
        {
            "presetUsed": "photo",
            "qualityMode": "balanced",
            "pipelineMode": "photo",
            "outlineBackend": "controlnet",
            "controlnet": {"preprocessor": "canny"},
            "analysis": {
                "scores": {"photo": 0.81, "logo": 0.19},
                "background_type": "dark",
                "image_complexity": 0.42,
            },
        }
    )

    assert "Outline: ControlNet Canny" in panel.status_label.text()
