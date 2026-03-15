from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

try:
    import onnxruntime as ort
except ImportError:  # pragma: no cover - runtime optional
    ort = None

from .types import LoadedImage, PipelineConfig, PreprocessResult


@dataclass(slots=True)
class RasterArtworkAnalysis:
    is_candidate: bool
    foreground_mask: np.ndarray
    white_background_ratio: float
    corner_white_ratio: float
    foreground_ratio: float
    mean_saturation: float
    white_cutoff: int
    coarse_color_bins: int
    edge_density: float
    max_side: int


def load_image_data(path: str | Path) -> LoadedImage:
    image = Image.open(path)
    image = ImageOps.exif_transpose(image)

    alpha_mask: np.ndarray | None = None
    has_alpha = "A" in image.getbands()
    transparent_ratio = 0.0

    if has_alpha:
        rgba = np.asarray(image.convert("RGBA"), dtype=np.uint8)
        rgb = rgba[:, :, :3].astype(np.float32)
        alpha = rgba[:, :, 3].astype(np.float32) / 255.0

        white = np.full_like(rgb, 255.0)
        composited_rgb = rgb * alpha[..., None] + white * (1.0 - alpha[..., None])
        composited_rgb = np.clip(composited_rgb, 0, 255).astype(np.uint8)

        alpha_raw = rgba[:, :, 3].astype(np.uint8)
        alpha_mask = np.where(alpha_raw > 0, 255, 0).astype(np.uint8)
        transparent_ratio = float(np.mean(alpha_raw < 250))
    else:
        composited_rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)

    bgr = cv2.cvtColor(composited_rgb, cv2.COLOR_RGB2BGR)
    return LoadedImage(
        bgr=bgr,
        alpha_mask=alpha_mask,
        has_alpha=has_alpha,
        transparent_ratio=round(float(transparent_ratio), 4),
    )


def load_image_bgr(path: str | Path) -> np.ndarray:
    return load_image_data(path).bgr


def resize_to_max_side(image: np.ndarray, max_side_px: int) -> tuple[np.ndarray, float]:
    height, width = image.shape[:2]
    max_side = max(height, width)
    if max_side <= max_side_px:
        return image, 1.0
    scale = max_side_px / float(max_side)
    target = (max(1, int(width * scale)), max(1, int(height * scale)))
    return cv2.resize(image, target, interpolation=cv2.INTER_AREA), scale


def resize_loaded_image(loaded: LoadedImage, max_side_px: int) -> tuple[LoadedImage, float]:
    resized_bgr, scale = resize_to_max_side(loaded.bgr, max_side_px)
    if loaded.alpha_mask is None:
        return LoadedImage(
            bgr=resized_bgr,
            alpha_mask=None,
            has_alpha=loaded.has_alpha,
            transparent_ratio=loaded.transparent_ratio,
        ), scale

    resized_alpha = cv2.resize(
        loaded.alpha_mask,
        (resized_bgr.shape[1], resized_bgr.shape[0]),
        interpolation=cv2.INTER_NEAREST,
    )
    transparent_ratio = float(np.mean(resized_alpha < 250))
    return LoadedImage(
        bgr=resized_bgr,
        alpha_mask=resized_alpha,
        has_alpha=loaded.has_alpha,
        transparent_ratio=round(transparent_ratio, 4),
    ), scale


def compose_alpha_preview(bgr: np.ndarray, alpha_mask: np.ndarray | None) -> np.ndarray:
    if alpha_mask is None:
        return bgr

    height, width = bgr.shape[:2]
    tile = 20
    yy, xx = np.indices((height, width))
    checker = (((xx // tile) + (yy // tile)) % 2).astype(np.uint8)
    background = np.where(checker[..., None] == 0, 215, 240).astype(np.uint8)

    alpha = np.clip(alpha_mask.astype(np.float32) / 255.0, 0.0, 1.0)
    composed = bgr.astype(np.float32) * alpha[..., None] + background.astype(np.float32) * (1.0 - alpha[..., None])
    return np.clip(composed, 0, 255).astype(np.uint8)


def compose_segment_preview(quantized: np.ndarray, edge_map: np.ndarray) -> np.ndarray:
    preview = quantized.copy()
    if edge_map.ndim == 2:
        mask = edge_map > 0
        preview[mask] = (30, 30, 225)
        return preview
    return preview


def analyze_raster_artwork(image: np.ndarray, config: PipelineConfig) -> RasterArtworkAnalysis:
    height, width = image.shape[:2]
    max_side = max(height, width)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mean_saturation = float(np.mean(hsv[:, :, 1]))
    edge_map = cv2.Canny(gray, max(24, config.canny_low - 30), max(72, config.canny_high - 60))
    edge_density = float(np.mean(edge_map > 0))

    small = cv2.resize(image, (96, 96), interpolation=cv2.INTER_AREA)
    coarse_color_bins = int(len(np.unique((small // 32).reshape(-1, 3), axis=0)))

    min_channel = np.min(image, axis=2)
    corner_h = max(24, int(height * 0.12))
    corner_w = max(24, int(width * 0.12))
    corner_values = np.concatenate(
        [
            min_channel[:corner_h, :corner_w].reshape(-1),
            min_channel[:corner_h, width - corner_w :].reshape(-1),
            min_channel[height - corner_h :, :corner_w].reshape(-1),
            min_channel[height - corner_h :, width - corner_w :].reshape(-1),
        ]
    )

    estimated_white = float(np.percentile(corner_values, 60))
    white_cutoff = int(np.clip(estimated_white - 8.0, config.raster_artwork_white_cutoff_floor, 250))
    near_white = min_channel >= white_cutoff

    white_background_ratio = float(np.mean(near_white))
    corner_white_ratio = float(np.mean(corner_values >= white_cutoff))

    foreground_mask = np.where(near_white, 0, 255).astype(np.uint8)
    darkness_mask = np.where(gray < min(white_cutoff, 245), 255, 0).astype(np.uint8)
    foreground_mask = cv2.bitwise_or(foreground_mask, darkness_mask)
    foreground_mask = cv2.morphologyEx(
        foreground_mask,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        iterations=1,
    )
    foreground_mask = cv2.morphologyEx(
        foreground_mask,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        iterations=1,
    )

    min_area = max(12, int(height * width * 0.000006))
    foreground_mask = _remove_small_components(foreground_mask, min_area)
    foreground_ratio = float(np.count_nonzero(foreground_mask)) / float(foreground_mask.size)

    is_base_candidate = (
        white_background_ratio >= float(config.raster_artwork_white_ratio_threshold)
        and corner_white_ratio >= float(config.raster_artwork_corner_ratio_threshold)
        and float(config.raster_artwork_min_foreground_ratio)
        <= foreground_ratio
        <= float(config.raster_artwork_max_foreground_ratio)
        and (
            mean_saturation <= float(config.raster_artwork_saturation_threshold)
            or coarse_color_bins <= int(config.raster_artwork_color_bins_threshold)
        )
    )
    is_small_image_candidate = (
        max_side <= int(config.raster_artwork_small_image_max_side)
        and white_background_ratio >= float(config.raster_artwork_white_ratio_threshold)
        and corner_white_ratio >= float(config.raster_artwork_corner_ratio_threshold)
        and float(config.raster_artwork_min_foreground_ratio)
        <= foreground_ratio
        <= float(config.raster_artwork_max_foreground_ratio)
        and coarse_color_bins <= int(config.raster_artwork_color_bins_threshold)
        and edge_density <= float(config.raster_artwork_edge_density_threshold)
    )
    is_candidate = is_base_candidate or is_small_image_candidate

    return RasterArtworkAnalysis(
        is_candidate=is_candidate,
        foreground_mask=foreground_mask,
        white_background_ratio=round(white_background_ratio, 4),
        corner_white_ratio=round(corner_white_ratio, 4),
        foreground_ratio=round(foreground_ratio, 4),
        mean_saturation=round(mean_saturation, 4),
        white_cutoff=int(white_cutoff),
        coarse_color_bins=int(coarse_color_bins),
        edge_density=round(edge_density, 4),
        max_side=int(max_side),
    )


def preprocess_for_outline(image: np.ndarray, config: PipelineConfig, preset_used: str = "photo") -> PreprocessResult:
    settings = config.settings.clamped()
    detail_ratio = settings.detail / 100.0
    smooth_ratio = settings.smoothness / 100.0
    outline_ratio = settings.outline_strength / 100.0
    quality_mode = config.validated_quality_mode()

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe_limit = 2.8 + detail_ratio * 1.2 + (0.3 if quality_mode == "high_quality" else 0.0)
    clahe = cv2.createCLAHE(clipLimit=clahe_limit, tileGridSize=(8, 8))
    enhanced_gray = clahe.apply(gray)
    blur_kernel = 5 if smooth_ratio >= 0.2 else 3
    if preset_used in {"logo", "text_typo"}:
        blur_kernel = 3
    if preset_used == "sticker_cutting":
        blur_kernel = 5
    blurred = cv2.GaussianBlur(enhanced_gray, (blur_kernel, blur_kernel), 0)
    if preset_used == "text_typo":
        outline_ratio = max(outline_ratio, 0.72)
    if outline_ratio > 0.45:
        sharpen_kernel = np.array([[0, -1, 0], [-1, 5.0 + outline_ratio, -1], [0, -1, 0]], dtype=np.float32)
        blurred = cv2.filter2D(blurred, -1, sharpen_kernel)

    block_size = max(3, config.adaptive_block_size | 1)
    adaptive_c = max(1, int(config.adaptive_c + (smooth_ratio - detail_ratio) * 4))
    adaptive = cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        block_size,
        adaptive_c,
    )
    if preset_used == "text_typo":
        canny_low = max(12, int(config.canny_low - detail_ratio * 34))
        canny_high = max(canny_low + 14, int(config.canny_high - detail_ratio * 42 + smooth_ratio * 10))
    else:
        canny_low = max(16, int(config.canny_low - detail_ratio * 22))
        canny_high = max(canny_low + 18, int(config.canny_high - detail_ratio * 28 + smooth_ratio * 18))
    canny = cv2.Canny(blurred, canny_low, canny_high)
    edge_map = cv2.bitwise_or(canny, adaptive)
    if quality_mode == "high_quality":
        lap = cv2.Laplacian(blurred, cv2.CV_8U)
        edge_map = cv2.bitwise_or(edge_map, cv2.threshold(lap, 18, 255, cv2.THRESH_BINARY)[1])

    threshold_map = cv2.morphologyEx(
        adaptive,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3 if smooth_ratio < 0.55 else 5, 3 if smooth_ratio < 0.55 else 5)),
        iterations=1,
    )
    foreground_mask = cv2.morphologyEx(
        threshold_map,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1 if detail_ratio > 0.65 else 3, 1 if detail_ratio > 0.65 else 3)),
        iterations=1,
    )
    if preset_used == "dtf_screen_print":
        foreground_mask = cv2.morphologyEx(
            foreground_mask,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
            iterations=1,
        )
    if preset_used == "sticker_cutting":
        foreground_mask = cv2.morphologyEx(
            foreground_mask,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
            iterations=1,
        )
        threshold_map = cv2.morphologyEx(
            threshold_map,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
            iterations=1,
        )

    return PreprocessResult(
        gray=gray,
        enhanced_gray=enhanced_gray,
        edge_map=edge_map,
        threshold_map=threshold_map,
        foreground_mask=foreground_mask,
    )


def preprocess_ai_outline(
    image: np.ndarray,
    config: PipelineConfig,
    preset_used: str = "photo",
) -> PreprocessResult:
    settings = config.settings.clamped()
    detail_ratio = settings.detail / 100.0
    smooth_ratio = settings.smoothness / 100.0

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if float(np.mean(gray)) < 128.0:
        gray = cv2.bitwise_not(gray)

    clahe = cv2.createCLAHE(clipLimit=2.0 + detail_ratio * 0.6, tileGridSize=(8, 8))
    enhanced_gray = clahe.apply(gray)
    blur_size = 3 if smooth_ratio < 0.65 else 5
    blurred = cv2.GaussianBlur(enhanced_gray, (blur_size, blur_size), 0)

    _, otsu = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    adaptive = cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        max(3, config.adaptive_block_size | 1),
        max(1, int(config.adaptive_c - smooth_ratio * 2)),
    )
    line_map = cv2.bitwise_or(otsu, adaptive)

    close_kernel = 3 if preset_used in {"logo", "text_typo"} else 5
    line_map = cv2.morphologyEx(
        line_map,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_kernel, close_kernel)),
        iterations=1,
    )
    line_map = cv2.morphologyEx(
        line_map,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        iterations=1,
    )

    min_area = max(12, int(image.shape[0] * image.shape[1] * 0.00002))
    filled_mask = _fill_regions_from_edges(line_map, min_area=min_area)
    filled_mask = cv2.morphologyEx(
        filled_mask,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        iterations=1,
    )
    filled_mask = _remove_small_components(filled_mask, min_area)

    edge_map = cv2.Canny(blurred, 20, 90)
    edge_map = cv2.bitwise_or(edge_map, line_map)
    edge_map = cv2.bitwise_and(edge_map, cv2.dilate(filled_mask, np.ones((3, 3), dtype=np.uint8), iterations=1))

    return PreprocessResult(
        gray=gray,
        enhanced_gray=enhanced_gray,
        edge_map=edge_map,
        threshold_map=filled_mask,
        foreground_mask=filled_mask,
    )


def preprocess_artwork(
    image: np.ndarray,
    alpha_mask: np.ndarray,
    config: PipelineConfig,
    preset_used: str = "logo",
) -> PreprocessResult:
    settings = config.settings.clamped()
    detail_ratio = settings.detail / 100.0
    smooth_ratio = settings.smoothness / 100.0

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.1 + detail_ratio * 0.8, tileGridSize=(8, 8))
    enhanced_gray = clahe.apply(gray)
    blur_size = 3 if smooth_ratio < 0.65 or preset_used in {"logo", "text_typo"} else 5
    if preset_used == "sticker_cutting":
        blur_size = 5
    blurred = cv2.GaussianBlur(enhanced_gray, (blur_size, blur_size), 0)

    alpha_foreground = np.where(alpha_mask > 0, 255, 0).astype(np.uint8)
    alpha_foreground = cv2.morphologyEx(
        alpha_foreground,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        iterations=1,
    )
    alpha_foreground = cv2.morphologyEx(
        alpha_foreground,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        iterations=1,
    )

    block_size = max(3, config.adaptive_block_size | 1)
    adaptive_c = max(1, int(config.adaptive_c + (smooth_ratio - detail_ratio) * 4))
    adaptive = cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        block_size,
        adaptive_c,
    )
    adaptive = cv2.bitwise_and(adaptive, alpha_foreground)

    if preset_used == "text_typo":
        color_edge = cv2.Canny(
            blurred,
            max(10, int(config.canny_low - detail_ratio * 36)),
            max(48, int(config.canny_high - detail_ratio * 44 + smooth_ratio * 8)),
        )
    else:
        color_edge = cv2.Canny(
            blurred,
            max(20, int(config.canny_low - detail_ratio * 26)),
            max(68, int(config.canny_high - detail_ratio * 28 + smooth_ratio * 10)),
        )
    color_edge = cv2.bitwise_and(color_edge, alpha_foreground)
    alpha_edge = cv2.Canny(alpha_foreground, 20, 80)
    edge_map = cv2.bitwise_or(cv2.bitwise_or(color_edge, alpha_edge), adaptive)
    edge_map = cv2.morphologyEx(
        edge_map,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        iterations=1,
    )
    if preset_used == "sticker_cutting":
        alpha_foreground = cv2.morphologyEx(
            alpha_foreground,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
            iterations=1,
        )
        edge_map = cv2.Canny(alpha_foreground, 16, 72)

    return PreprocessResult(
        gray=gray,
        enhanced_gray=enhanced_gray,
        edge_map=edge_map,
        threshold_map=adaptive,
        foreground_mask=alpha_foreground,
    )


def preprocess_raster_artwork(
    image: np.ndarray,
    foreground_mask: np.ndarray,
    config: PipelineConfig,
    preset_used: str = "illustration",
) -> PreprocessResult:
    settings = config.settings.clamped()
    detail_ratio = settings.detail / 100.0
    smooth_ratio = settings.smoothness / 100.0
    corners_ratio = settings.corners / 100.0
    quality_mode = config.validated_quality_mode()

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe_limit = 2.1 + detail_ratio * 1.0 + (0.25 if quality_mode == "high_quality" else 0.0)
    clahe = cv2.createCLAHE(clipLimit=clahe_limit, tileGridSize=(8, 8))
    enhanced_gray = clahe.apply(gray)
    blur_size = 3 if smooth_ratio < 0.7 or preset_used in {"logo", "text_typo"} else 5
    if preset_used == "sticker_cutting":
        blur_size = 5
    blurred = cv2.GaussianBlur(enhanced_gray, (blur_size, blur_size), 0)
    if corners_ratio >= 0.7:
        blurred = cv2.addWeighted(enhanced_gray, 0.2, blurred, 0.8, 0)

    cleaned_foreground = np.where(foreground_mask > 0, 255, 0).astype(np.uint8)
    cleaned_foreground = cv2.morphologyEx(
        cleaned_foreground,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        iterations=1,
    )
    cleaned_foreground = cv2.morphologyEx(
        cleaned_foreground,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        iterations=1,
    )

    block_size = max(3, config.adaptive_block_size | 1)
    adaptive_c = max(1, int(config.adaptive_c + (smooth_ratio - detail_ratio) * 4))
    adaptive = cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        block_size,
        adaptive_c,
    )
    _, otsu = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    threshold_map = cv2.bitwise_or(adaptive, otsu)
    threshold_map = cv2.bitwise_and(threshold_map, cleaned_foreground)
    if quality_mode == "high_quality":
        lap = cv2.Laplacian(blurred, cv2.CV_8U)
        detail_map = cv2.threshold(lap, 16, 255, cv2.THRESH_BINARY)[1]
        threshold_map = cv2.bitwise_or(threshold_map, cv2.bitwise_and(detail_map, cleaned_foreground))

    if preset_used == "text_typo":
        color_edge = cv2.Canny(
            blurred,
            max(8, int(config.canny_low - detail_ratio * 42)),
            max(42, int(config.canny_high - detail_ratio * 82 + smooth_ratio * 10)),
        )
    else:
        color_edge = cv2.Canny(
            blurred,
            max(14, int(config.canny_low - detail_ratio * 34)),
            max(58, int(config.canny_high - detail_ratio * 72 + smooth_ratio * 18)),
        )
    color_edge = cv2.bitwise_and(color_edge, cv2.dilate(cleaned_foreground, np.ones((3, 3), dtype=np.uint8), iterations=1))
    foreground_edge = cv2.Canny(cleaned_foreground, 20, 80)
    edge_map = cv2.bitwise_or(color_edge, foreground_edge)
    edge_map = cv2.bitwise_or(edge_map, threshold_map)
    if preset_used == "dtf_screen_print":
        edge_map = cv2.morphologyEx(
            edge_map,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
            iterations=1,
        )
    if preset_used == "sticker_cutting":
        cleaned_foreground = cv2.morphologyEx(
            cleaned_foreground,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
            iterations=1,
        )
        threshold_map = cv2.morphologyEx(
            threshold_map,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
            iterations=1,
        )
        edge_map = cv2.Canny(cleaned_foreground, 14, 64)
    edge_map = cv2.morphologyEx(
        edge_map,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        iterations=1,
    )

    return PreprocessResult(
        gray=gray,
        enhanced_gray=enhanced_gray,
        edge_map=edge_map,
        threshold_map=threshold_map,
        foreground_mask=cleaned_foreground,
    )


def boost_trace_inputs(
    image: np.ndarray,
    mask: np.ndarray | None,
    config: PipelineConfig,
) -> tuple[np.ndarray, np.ndarray | None, dict[str, object]]:
    height, width = image.shape[:2]
    max_side = max(height, width)
    settings = config.settings.clamped()
    trigger = int(config.small_trace_boost_trigger_px)
    if max_side >= trigger:
        return image, mask, {
            "applied": False,
            "scale": 1.0,
            "sourceMaxSide": int(max_side),
            "targetMaxSide": int(max_side),
        }

    target_side = int(
        config.small_trace_target_px_high_quality
        if config.validated_quality_mode() == "high_quality"
        else config.small_trace_target_px
    )
    target_side = max(trigger, target_side)
    scale = min(float(config.small_trace_max_scale), target_side / float(max_side))
    if scale <= 1.05:
        return image, mask, {
            "applied": False,
            "scale": 1.0,
            "sourceMaxSide": int(max_side),
            "targetMaxSide": int(max_side),
        }

    target_width = max(1, int(round(width * scale)))
    target_height = max(1, int(round(height * scale)))
    boosted = cv2.resize(image, (target_width, target_height), interpolation=cv2.INTER_LANCZOS4)
    sigma = 0.55 + (settings.smoothness / 100.0) * 0.6
    softened = cv2.GaussianBlur(boosted, (0, 0), sigma)
    sharpen_gain = 0.08 + (settings.detail / 100.0) * 0.05
    boosted = cv2.addWeighted(boosted, 1.0 + sharpen_gain, softened, -sharpen_gain, 0)

    boosted_mask: np.ndarray | None = None
    if mask is not None:
        resized_mask = cv2.resize(mask, (target_width, target_height), interpolation=cv2.INTER_LINEAR)
        boosted_mask = np.where(resized_mask > 16, 255, 0).astype(np.uint8)

    return boosted, boosted_mask, {
        "applied": True,
        "scale": round(float(scale), 4),
        "sourceMaxSide": int(max_side),
        "targetMaxSide": int(max(target_height, target_width)),
    }


@dataclass(slots=True)
class _OrtLayout:
    is_nchw: bool
    channels: int
    height: int
    width: int
    dynamic_spatial: bool = False


class RealESRGANRestorer:
    def __init__(self, model_path: Path):
        self.model_path = model_path
        self.session: ort.InferenceSession | None = None
        self.runtime_warning: str | None = None
        self.backend = "classical"
        self._layout: _OrtLayout | None = None
        self._input_name: str | None = None
        self._output_name: str | None = None
        self._scale: int = 4
        self._max_process_side: int = 64

        if ort is None:
            self.runtime_warning = "onnxruntime is not installed. Real-ESRGAN runs in classical fallback mode."
            return
        if not model_path.exists() or model_path.stat().st_size < 1024:
            self.runtime_warning = f"Model not ready: {model_path.name}. Using classical restoration fallback."
            return

        try:
            providers = ["CPUExecutionProvider"]
            self.session = ort.InferenceSession(str(model_path), providers=providers)
            input_meta = self.session.get_inputs()[0]
            output_meta = self.session.get_outputs()[0]

            self._layout = _resolve_layout(input_meta.shape)
            self._input_name = input_meta.name
            self._output_name = output_meta.name
            self._scale = _estimate_upscale_factor(self._layout, output_meta.shape)
            self.backend = "onnx"
        except Exception as error:  # pragma: no cover - runtime dependency
            self.runtime_warning = f"Failed to load Real-ESRGAN ONNX: {error}. Falling back to classical mode."
            self.session = None
            self.backend = "classical"

    @property
    def loaded(self) -> bool:
        return self.session is not None

    def restore(self, image: np.ndarray) -> tuple[np.ndarray, dict[str, object]]:
        if self.session is None:
            return self._classical_restore(image), {"backend": self.backend, "fallback": True}

        try:
            assert self._layout is not None
            assert self._input_name is not None
            assert self._output_name is not None

            restored, tile_count = self._restore_tiled(
                image=image,
                layout=self._layout,
                input_name=self._input_name,
                output_name=self._output_name,
            )
            return restored, {
                "backend": self.backend,
                "fallback": False,
                "tile_count": int(tile_count),
                "upscale_factor": int(self._scale),
            }
        except Exception as error:  # pragma: no cover - runtime dependency
            self.runtime_warning = f"Real-ESRGAN inference failed: {error}. Using classical fallback."
            return self._classical_restore(image), {"backend": "classical", "fallback": True}

    def _restore_tiled(
        self,
        image: np.ndarray,
        layout: _OrtLayout,
        input_name: str,
        output_name: str,
    ) -> tuple[np.ndarray, int]:
        assert self.session is not None

        original_h, original_w = image.shape[:2]
        max_side = max(original_h, original_w)
        if max_side > self._max_process_side:
            scale = self._max_process_side / float(max_side)
            work_w = max(1, int(original_w * scale))
            work_h = max(1, int(original_h * scale))
            work = cv2.resize(image, (work_w, work_h), interpolation=cv2.INTER_AREA)
        else:
            work = image

        tile_h = max(16, int(layout.height))
        tile_w = max(16, int(layout.width))

        padded, pad_h, pad_w = _pad_to_tile(work, tile_h, tile_w)
        sr_padded = np.zeros((pad_h * self._scale, pad_w * self._scale, 3), dtype=np.uint8)

        tile_count = 0
        for y in range(0, pad_h, tile_h):
            for x in range(0, pad_w, tile_w):
                tile = padded[y : y + tile_h, x : x + tile_w]
                network_input = _prepare_input(tile, layout)
                output = self.session.run([output_name], {input_name: network_input})[0]
                sr_tile = _decode_output_image(output)

                expected_h = tile_h * self._scale
                expected_w = tile_w * self._scale
                if sr_tile.shape[0] != expected_h or sr_tile.shape[1] != expected_w:
                    sr_tile = cv2.resize(sr_tile, (expected_w, expected_h), interpolation=cv2.INTER_CUBIC)

                sy0 = y * self._scale
                sy1 = (y + tile_h) * self._scale
                sx0 = x * self._scale
                sx1 = (x + tile_w) * self._scale
                sr_padded[sy0:sy1, sx0:sx1] = sr_tile
                tile_count += 1

        work_h, work_w = work.shape[:2]
        sr_work = sr_padded[: work_h * self._scale, : work_w * self._scale]
        restored = cv2.resize(sr_work, (original_w, original_h), interpolation=cv2.INTER_CUBIC)
        blended = cv2.addWeighted(image, 0.45, restored, 0.55, 0)
        return blended, tile_count

    def _classical_restore(self, image: np.ndarray) -> np.ndarray:
        denoised = cv2.fastNlMeansDenoisingColored(image, None, 3, 3, 7, 21)
        sharpen_kernel = np.array([[0, -1, 0], [-1, 5.2, -1], [0, -1, 0]], dtype=np.float32)
        sharpened = cv2.filter2D(denoised, -1, sharpen_kernel)
        return cv2.addWeighted(denoised, 0.6, sharpened, 0.4, 0)


def _resolve_layout(shape: list[object]) -> _OrtLayout:
    if len(shape) != 4:
        return _OrtLayout(is_nchw=True, channels=3, height=128, width=128, dynamic_spatial=False)

    dims: list[int | None] = [dim if isinstance(dim, int) and dim > 0 else None for dim in shape]
    dynamic_spatial = dims[2] is None or dims[3] is None

    n, d1, d2, d3 = dims
    del n
    if d1 in (1, 3):
        return _OrtLayout(
            is_nchw=True,
            channels=d1,
            height=d2 or 128,
            width=d3 or 128,
            dynamic_spatial=dynamic_spatial,
        )
    if d3 in (1, 3):
        return _OrtLayout(
            is_nchw=False,
            channels=d3,
            height=d1 or 128,
            width=d2 or 128,
            dynamic_spatial=dynamic_spatial,
        )
    return _OrtLayout(
        is_nchw=True,
        channels=3,
        height=d2 or 128,
        width=d3 or 128,
        dynamic_spatial=dynamic_spatial,
    )


def _estimate_upscale_factor(layout: _OrtLayout, output_shape: list[object]) -> int:
    if len(output_shape) != 4:
        return 4

    dims = [dim if isinstance(dim, int) and dim > 0 else None for dim in output_shape]
    _, d1, d2, d3 = dims

    if layout.is_nchw:
        out_h = d2
        out_w = d3
    else:
        out_h = d1
        out_w = d2

    if out_h is None or out_w is None:
        return 4

    scale_h = max(1, int(round(out_h / max(1, layout.height))))
    scale_w = max(1, int(round(out_w / max(1, layout.width))))
    return max(1, min(scale_h, scale_w))


def _prepare_input(image: np.ndarray, layout: _OrtLayout) -> np.ndarray:
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (layout.width, layout.height), interpolation=cv2.INTER_CUBIC)

    if layout.channels == 1:
        resized = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY)[..., None]

    normalized = resized.astype(np.float32) / 255.0

    if layout.is_nchw:
        return np.transpose(normalized, (2, 0, 1))[None, ...]
    return normalized[None, ...]


def _pad_to_tile(image: np.ndarray, tile_h: int, tile_w: int) -> tuple[np.ndarray, int, int]:
    height, width = image.shape[:2]
    pad_h = ((height + tile_h - 1) // tile_h) * tile_h
    pad_w = ((width + tile_w - 1) // tile_w) * tile_w
    if pad_h == height and pad_w == width:
        return image, pad_h, pad_w

    padded = cv2.copyMakeBorder(
        image,
        0,
        pad_h - height,
        0,
        pad_w - width,
        borderType=cv2.BORDER_REFLECT_101,
    )
    return padded, pad_h, pad_w


def _fill_regions_from_edges(edge_map: np.ndarray, min_area: int) -> np.ndarray:
    contours, hierarchy = cv2.findContours(edge_map, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None:
        return edge_map.copy()

    filled = np.zeros_like(edge_map)
    tree = hierarchy[0]
    for index, contour in enumerate(contours):
        if cv2.contourArea(contour) < float(min_area):
            continue

        depth = 0
        parent = int(tree[index][3])
        while parent != -1:
            depth += 1
            parent = int(tree[parent][3])

        color = 255 if depth % 2 == 0 else 0
        cv2.drawContours(filled, contours, index, color, thickness=-1)

    return cv2.bitwise_or(filled, edge_map)


def _decode_output_image(output: np.ndarray) -> np.ndarray:
    array = np.asarray(output)
    if array.ndim == 4:
        if array.shape[1] in (1, 3):
            array = np.transpose(array[0], (1, 2, 0))
        elif array.shape[-1] in (1, 3):
            array = array[0]
        else:
            array = array[0, 0]
    elif array.ndim == 3:
        array = array
    elif array.ndim == 2:
        array = array[..., None]

    array = np.clip(array, 0.0, 1.0 if array.max() <= 1.0 else 255.0)
    if array.max() <= 1.0:
        array = array * 255.0
    array = array.astype(np.uint8)

    if array.ndim == 2:
        array = cv2.cvtColor(array, cv2.COLOR_GRAY2RGB)
    elif array.shape[2] == 1:
        array = cv2.cvtColor(array, cv2.COLOR_GRAY2RGB)

    return cv2.cvtColor(array, cv2.COLOR_RGB2BGR)


def _remove_small_components(mask: np.ndarray, min_area: int) -> np.ndarray:
    components, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if components <= 1:
        return mask

    cleaned = np.zeros_like(mask)
    for index in range(1, components):
        if int(stats[index, cv2.CC_STAT_AREA]) >= int(min_area):
            cleaned[labels == index] = 255
    return cleaned
