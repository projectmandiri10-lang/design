from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np


ColorCount = int
TracingPreset = Literal[
    "auto",
    "logo",
    "illustration",
    "text_typo",
    "dtf_screen_print",
    "sticker_cutting",
    "sablon",
    "photo",
]
ResolvedPreset = Literal["logo", "illustration", "text_typo", "dtf_screen_print", "sticker_cutting", "photo"]
TraceQualityMode = Literal["balanced", "high_quality"]
BackgroundMode = Literal["transparent", "keep_white", "drop_white"]
ControlNetPreprocessor = Literal["lineart", "canny"]


@dataclass(slots=True)
class PipelinePaths:
    detector_model: Path
    realesrgan_model: Path
    unet_model: Path
    potrace_bin: Path | None = None
    inkscape_bin: Path | None = None


@dataclass(slots=True)
class AdvancedTraceSettings:
    detail: int = 70
    smoothness: int = 35
    corners: int = 75
    despeckle: int = 35
    min_shape_area: int = 18
    small_image_boost: bool = True
    ignore_white: bool = False
    outline_strength: int = 55
    cutline_offset: int = 8

    def clamped(self) -> AdvancedTraceSettings:
        return AdvancedTraceSettings(
            detail=_clamp_int(self.detail, 0, 100),
            smoothness=_clamp_int(self.smoothness, 0, 100),
            corners=_clamp_int(self.corners, 0, 100),
            despeckle=_clamp_int(self.despeckle, 0, 100),
            min_shape_area=_clamp_int(self.min_shape_area, 1, 10000),
            small_image_boost=bool(self.small_image_boost),
            ignore_white=bool(self.ignore_white),
            outline_strength=_clamp_int(self.outline_strength, 0, 100),
            cutline_offset=_clamp_int(self.cutline_offset, 0, 128),
        )

    def signature(self) -> tuple[object, ...]:
        value = self.clamped()
        return (
            value.detail,
            value.smoothness,
            value.corners,
            value.despeckle,
            value.min_shape_area,
            value.small_image_boost,
            value.ignore_white,
            value.outline_strength,
            value.cutline_offset,
        )


@dataclass(slots=True)
class ControlNetSettings:
    enabled: bool = False
    base_url: str = "http://127.0.0.1:7860"
    preprocessor: ControlNetPreprocessor = "lineart"
    connect_timeout_s: float = 2.5
    request_timeout_s: float = 60.0

    def validated_preprocessor(self) -> ControlNetPreprocessor:
        allowed: tuple[ControlNetPreprocessor, ...] = ("lineart", "canny")
        return self.preprocessor if self.preprocessor in allowed else "lineart"

    def normalized_base_url(self) -> str:
        value = str(self.base_url or "http://127.0.0.1:7860").strip()
        if not value:
            value = "http://127.0.0.1:7860"
        return value.rstrip("/")

    def resolved_model_prefix(self) -> str:
        mapping: dict[ControlNetPreprocessor, str] = {
            "lineart": "control_v11p_sd15_lineart",
            "canny": "control_v11p_sd15_canny",
        }
        return mapping[self.validated_preprocessor()]

    def signature(self) -> tuple[object, ...]:
        return (
            bool(self.enabled),
            self.normalized_base_url(),
            self.validated_preprocessor(),
            round(_clamp_float(self.connect_timeout_s, 0.25, 30.0), 3),
            round(_clamp_float(self.request_timeout_s, 1.0, 300.0), 3),
        )


@dataclass(slots=True)
class PipelineConfig:
    paths: PipelinePaths
    color_count: ColorCount = 4
    max_side_px: int = 2048
    speed_mode: str = "balanced"
    preset: TracingPreset = "auto"
    quality_mode: TraceQualityMode = "balanced"
    background_mode: BackgroundMode = "transparent"
    settings: AdvancedTraceSettings = field(default_factory=AdvancedTraceSettings)
    controlnet: ControlNetSettings = field(default_factory=ControlNetSettings)
    canny_low: int = 70
    canny_high: int = 180
    adaptive_block_size: int = 31
    adaptive_c: int = 4
    artwork_alpha_threshold: float = 0.01
    raster_artwork_white_ratio_threshold: float = 0.55
    raster_artwork_corner_ratio_threshold: float = 0.72
    raster_artwork_saturation_threshold: float = 28.0
    raster_artwork_min_foreground_ratio: float = 0.01
    raster_artwork_max_foreground_ratio: float = 0.45
    raster_artwork_white_cutoff_floor: int = 235
    raster_artwork_color_bins_threshold: int = 96
    raster_artwork_edge_density_threshold: float = 0.1
    raster_artwork_small_image_max_side: int = 768
    min_component_ratio_photo: float = 0.0002
    min_component_ratio_artwork: float = 0.00003
    min_component_ratio_raster_artwork: float = 0.00002
    small_trace_boost_trigger_px: int = 900
    small_trace_target_px: int = 1400
    small_trace_target_px_high_quality: int = 1800
    small_trace_max_scale: float = 16.0

    def validated_color_count(self) -> int:
        allowed = {1, 2, 4, 6, 8}
        if self.color_count in allowed:
            return self.color_count
        if self.color_count <= 1:
            return 1
        if self.color_count <= 2:
            return 2
        if self.color_count <= 4:
            return 4
        if self.color_count <= 6:
            return 6
        return 8

    def validated_preset(self) -> TracingPreset:
        allowed: tuple[TracingPreset, ...] = (
            "auto",
            "logo",
            "illustration",
            "text_typo",
            "dtf_screen_print",
            "sticker_cutting",
            "sablon",
            "photo",
        )
        return self.preset if self.preset in allowed else "auto"

    def validated_quality_mode(self) -> TraceQualityMode:
        allowed: tuple[TraceQualityMode, ...] = ("balanced", "high_quality")
        return self.quality_mode if self.quality_mode in allowed else "balanced"

    def validated_background_mode(self) -> BackgroundMode:
        allowed: tuple[BackgroundMode, ...] = ("transparent", "keep_white", "drop_white")
        return self.background_mode if self.background_mode in allowed else "transparent"

    def quality_multiplier(self) -> float:
        return 1.25 if self.validated_quality_mode() == "high_quality" else 1.0

    def settings_signature(self) -> tuple[object, ...]:
        return (
            self.validated_preset(),
            self.validated_quality_mode(),
            self.validated_background_mode(),
            self.validated_color_count(),
            *self.settings.clamped().signature(),
            *self.controlnet.signature(),
        )


@dataclass(slots=True)
class ModelRuntimeInfo:
    detector_loaded: bool = False
    realesrgan_loaded: bool = False
    unet_loaded: bool = False
    detector_backend: str = "contour"
    realesrgan_backend: str = "classical"
    unet_backend: str = "classical"
    warnings: list[str] = field(default_factory=list)


@dataclass(slots=True)
class VectorPathData:
    d: str
    transform: str | None = None


@dataclass(slots=True)
class LayerResult:
    index: int
    hex_color: str
    pixel_count: int
    mask: np.ndarray
    svg_paths: list[VectorPathData]
    node_count: int


@dataclass(slots=True)
class DetectionResult:
    cropped_image: np.ndarray
    bbox_xywh: tuple[int, int, int, int]
    confidence: float
    method: str
    mask: np.ndarray


@dataclass(slots=True)
class PreprocessResult:
    gray: np.ndarray
    enhanced_gray: np.ndarray
    edge_map: np.ndarray
    threshold_map: np.ndarray
    foreground_mask: np.ndarray


@dataclass(slots=True)
class LoadedImage:
    bgr: np.ndarray
    alpha_mask: np.ndarray | None
    has_alpha: bool
    transparent_ratio: float


@dataclass(slots=True)
class TraceAnalysis:
    scores: dict[str, float]
    recommended_preset: ResolvedPreset
    background_type: str
    image_complexity: float
    small_image: bool
    white_background_ratio: float
    edge_density: float
    coarse_color_bins: int
    texture_score: float
    alpha_transparency_ratio: float

    def to_metadata(self) -> dict[str, Any]:
        data = asdict(self)
        data["scores"] = {key: round(float(value), 4) for key, value in self.scores.items()}
        data["image_complexity"] = round(float(self.image_complexity), 4)
        data["white_background_ratio"] = round(float(self.white_background_ratio), 4)
        data["edge_density"] = round(float(self.edge_density), 4)
        data["texture_score"] = round(float(self.texture_score), 4)
        data["alpha_transparency_ratio"] = round(float(self.alpha_transparency_ratio), 4)
        return data


@dataclass(slots=True)
class ProcessResult:
    input_path: Path
    timings_ms: dict[str, float]
    warnings: list[str]
    runtime_info: ModelRuntimeInfo
    previews: dict[str, np.ndarray]
    layers: list[LayerResult]
    exports: dict[str, Path]
    palette: list[dict[str, Any]]
    vector_svg: str
    metadata: dict[str, Any] = field(default_factory=dict)


def _clamp_int(value: int, minimum: int, maximum: int) -> int:
    return max(minimum, min(maximum, int(value)))


def _clamp_float(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, float(value)))
