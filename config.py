from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from core.types import PipelineConfig, PipelinePaths


APP_NAME = "AUTO VECTOR SABLON AI"
ORGANIZATION_NAME = "Auto Vector Sablon AI"
DEFAULT_A1111_URL = "http://127.0.0.1:7860"
DEFAULT_OUTPUT_DIRNAME = "output"
DEFAULT_TEMP_DIRNAME = "temp"
DEFAULT_CLEANUP_PROMPT = (
    "clean vector logo, flat colors, screen printing design, bold outline, minimal colors, vector illustration"
)
DEFAULT_CLEANUP_NEGATIVE_PROMPT = "fabric texture, wrinkles, gradient, shadow, blur, noise"


@dataclass(frozen=True, slots=True)
class CleanupPromptConfig:
    prompt: str = DEFAULT_CLEANUP_PROMPT
    negative_prompt: str = DEFAULT_CLEANUP_NEGATIVE_PROMPT
    denoising_strength: float = 0.35
    steps: int = 20
    cfg_scale: float = 6.0
    connect_timeout_s: float = 2.5
    request_timeout_s: float = 90.0


@dataclass(frozen=True, slots=True)
class AppRuntimeConfig:
    app_name: str = APP_NAME
    organization_name: str = ORGANIZATION_NAME
    output_dirname: str = DEFAULT_OUTPUT_DIRNAME
    temp_dirname: str = DEFAULT_TEMP_DIRNAME
    cleanup: CleanupPromptConfig = CleanupPromptConfig()

    def output_dir(self, project_root: Path) -> Path:
        path = project_root / self.output_dirname
        path.mkdir(parents=True, exist_ok=True)
        return path

    def temp_dir(self, project_root: Path) -> Path:
        path = project_root / self.temp_dirname
        path.mkdir(parents=True, exist_ok=True)
        return path


def build_runtime_config() -> AppRuntimeConfig:
    return AppRuntimeConfig()


def build_default_pipeline_config(project_root: Path) -> PipelineConfig:
    model_dir = project_root / "models"
    paths = PipelinePaths(
        detector_model=model_dir / "sablon_detector.onnx",
        realesrgan_model=model_dir / "realesrgan_x4.onnx",
        unet_model=model_dir / "fabric_unet_denoise.onnx",
        potrace_bin=None,
        inkscape_bin=None,
    )
    config = PipelineConfig(
        paths=paths,
        color_count=6,
        preset="photo",
        quality_mode="balanced",
        background_mode="transparent",
    )
    config.controlnet.enabled = True
    config.controlnet.base_url = DEFAULT_A1111_URL
    config.controlnet.preprocessor = "lineart"
    config.controlnet.connect_timeout_s = 2.5
    config.controlnet.request_timeout_s = 90.0
    return config
