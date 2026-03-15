from __future__ import annotations

import cv2
import numpy as np

from .image_preprocess import RasterArtworkAnalysis, analyze_raster_artwork
from .types import LoadedImage, PipelineConfig, ResolvedPreset, TraceAnalysis, TracingPreset


def analyze_trace_input(loaded: LoadedImage, config: PipelineConfig) -> tuple[TraceAnalysis, RasterArtworkAnalysis]:
    image = loaded.bgr
    raster = analyze_raster_artwork(image, config)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edge_map = cv2.Canny(gray, max(20, config.canny_low - 20), max(70, config.canny_high - 40))
    edge_density = float(np.mean(edge_map > 0))
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    texture_score = float(np.clip(laplacian.var() / 3200.0, 0.0, 1.0))
    complexity = float(
        np.clip(
            0.34 * min(1.0, raster.coarse_color_bins / 140.0)
            + 0.28 * min(1.0, edge_density / 0.12)
            + 0.38 * texture_score,
            0.0,
            1.0,
        )
    )

    white_ratio = raster.white_background_ratio
    alpha_ratio = loaded.transparent_ratio if loaded.has_alpha else 0.0
    low_texture = 1.0 - texture_score
    low_complexity = 1.0 - complexity
    limited_colors = 1.0 - min(1.0, raster.coarse_color_bins / 120.0)
    small_image = bool(raster.max_side <= config.small_trace_boost_trigger_px)
    small_image_bonus = 1.0 if small_image else 0.0
    light_background = min(1.0, max(white_ratio, raster.corner_white_ratio))
    text_shape_bias = float(np.clip((0.14 - raster.foreground_ratio) / 0.14, 0.0, 1.0))
    text_like = float(np.clip(0.45 * text_shape_bias + 0.35 * limited_colors + 0.20 * low_texture, 0.0, 1.0))
    dtf_like = float(
        np.clip(
            0.36 * (1.0 - light_background)
            + 0.28 * limited_colors
            + 0.20 * low_texture
            + 0.16 * min(1.0, raster.coarse_color_bins / 80.0),
            0.0,
            1.0,
        )
    )
    sticker_color_bias = float(np.clip((24.0 - float(raster.coarse_color_bins)) / 24.0, 0.0, 1.0))
    sticker_edge_bias = float(np.clip((0.04 - edge_density) / 0.04, 0.0, 1.0))
    sticker_shape_bias = float(np.clip(1.0 - abs(raster.foreground_ratio - 0.25) / 0.25, 0.0, 1.0))
    sticker_like = float(
        np.clip(
            0.32 * max(alpha_ratio * 2.5, light_background)
            + 0.24 * sticker_color_bias
            + 0.18 * sticker_edge_bias
            + 0.14 * low_texture
            + 0.12 * sticker_shape_bias
            + 0.10 * min(1.0, alpha_ratio * 2.4)
            - 0.18 * text_like,
            0.0,
            1.0,
        )
    )

    scores: dict[str, float] = {
        "logo": np.clip(
            0.34 * max(alpha_ratio * 2.5, light_background)
            + 0.22 * limited_colors
            + 0.18 * low_texture
            + 0.14 * low_complexity
            + 0.12 * small_image_bonus
            - np.clip(text_like * 0.16 + sticker_like * sticker_color_bias * 0.18, 0.0, 0.28),
            0.0,
            1.0,
        ),
        "text_typo": np.clip(
            0.26 * max(alpha_ratio * 2.2, light_background)
            + 0.22 * limited_colors
            + 0.16 * min(1.0, edge_density / 0.05)
            + 0.14 * low_texture
            + 0.12 * small_image_bonus
            + 0.10 * text_shape_bias
            + 0.12 * text_like,
            0.0,
            1.0,
        )
        - np.clip(sticker_color_bias * sticker_edge_bias * 0.14, 0.0, 0.14),
        "illustration": np.clip(
            0.28 * max(alpha_ratio * 1.8, light_background)
            + 0.22 * min(1.0, raster.coarse_color_bins / 120.0)
            + 0.20 * low_texture
            + 0.18 * min(1.0, edge_density / 0.08)
            + 0.12 * (1.0 - abs(raster.foreground_ratio - 0.22) / 0.22),
            0.0,
            1.0,
        ),
        "dtf_screen_print": np.clip(
            0.28 * dtf_like
            + 0.22 * (1.0 - light_background)
            + 0.18 * min(1.0, raster.coarse_color_bins / 90.0)
            + 0.16 * min(1.0, edge_density / 0.05)
            + 0.16 * low_texture,
            0.0,
            1.0,
        ),
        "sticker_cutting": sticker_like,
        "photo": np.clip(
            0.38 * texture_score
            + 0.22 * complexity
            + 0.18 * min(1.0, raster.coarse_color_bins / 160.0)
            + 0.14 * (1.0 - light_background)
            + 0.08 * min(1.0, edge_density / 0.12),
            0.0,
            1.0,
        ),
    }

    recommended = max(scores.items(), key=lambda item: (item[1], _preset_rank(item[0])))[0]
    background_type = _background_type(loaded=loaded, raster=raster, texture_score=texture_score)

    analysis = TraceAnalysis(
        scores={key: round(float(value), 4) for key, value in scores.items()},
        recommended_preset=recommended,  # type: ignore[arg-type]
        background_type=background_type,
        image_complexity=round(complexity, 4),
        small_image=small_image,
        white_background_ratio=raster.white_background_ratio,
        edge_density=edge_density,
        coarse_color_bins=raster.coarse_color_bins,
        texture_score=round(texture_score, 4),
        alpha_transparency_ratio=alpha_ratio,
    )
    return analysis, raster


def resolve_preset(config_preset: TracingPreset, analysis: TraceAnalysis) -> ResolvedPreset:
    if config_preset == "sablon":
        return "dtf_screen_print"
    if config_preset == "auto":
        return analysis.recommended_preset
    return config_preset  # type: ignore[return-value]


def select_pipeline_mode(preset: ResolvedPreset, loaded: LoadedImage, analysis: TraceAnalysis) -> str:
    if loaded.has_alpha and loaded.alpha_mask is not None and loaded.transparent_ratio > 0.01:
        return "artwork"
    if preset in {"logo", "illustration", "text_typo", "dtf_screen_print", "sticker_cutting"} and analysis.background_type in {
        "transparent",
        "light",
    }:
        return "raster_artwork"
    return "photo"


def _preset_rank(name: str) -> int:
    order = {
        "sticker_cutting": 7,
        "text_typo": 6,
        "logo": 5,
        "illustration": 4,
        "dtf_screen_print": 3,
        "photo": 1,
    }
    return order.get(name, 0)


def _background_type(loaded: LoadedImage, raster: RasterArtworkAnalysis, texture_score: float) -> str:
    if loaded.has_alpha and loaded.transparent_ratio > 0.01:
        return "transparent"
    if raster.white_background_ratio >= 0.68 and raster.corner_white_ratio >= 0.8:
        return "light"
    if raster.white_background_ratio <= 0.2 and texture_score < 0.55:
        return "dark"
    return "textured"
