from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np

try:
    import potrace as py_potrace
except ImportError:  # pragma: no cover - optional
    py_potrace = None

from .types import AdvancedTraceSettings, LayerResult, TraceQualityMode, VectorPathData

SVG_NS = "http://www.w3.org/2000/svg"
ET.register_namespace("", SVG_NS)


def detect_potrace_binary(explicit_path: Path | None = None) -> Path | None:
    candidates: list[Path] = []
    if explicit_path is not None:
        candidates.append(explicit_path)

    env_bin = _safe_env_path("POTRACE_BIN")
    if env_bin is not None:
        candidates.append(env_bin)

    which = shutil.which("potrace")
    if which:
        candidates.append(Path(which))

    common_windows = [
        Path(r"C:\Program Files\potrace\potrace.exe"),
        Path(r"C:\Program Files (x86)\potrace\potrace.exe"),
        Path(__file__).resolve().parents[1] / "tools" / "potrace.exe",
        Path(__file__).resolve().parents[1] / "assets" / "tools" / "potrace" / "potrace.exe",
        Path(__file__).resolve().parents[1]
        / "assets"
        / "tools"
        / "potrace"
        / "potrace-1.16.win64"
        / "potrace.exe",
    ]
    candidates.extend(common_windows)

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _safe_env_path(name: str) -> Path | None:
    value = None
    try:
        value = __import__("os").environ.get(name)
    except Exception:  # pragma: no cover - defensive
        return None
    if not value:
        return None
    return Path(value)


def vectorize_by_color_layers(
    label_map: np.ndarray,
    palette: list[dict[str, object]],
    min_component_ratio: float,
    potrace_bin: Path | None,
    minimum_area_floor: int = 18,
    settings: AdvancedTraceSettings | None = None,
    quality_mode: TraceQualityMode = "balanced",
    preset: str = "auto",
) -> tuple[list[LayerResult], dict[str, object]]:
    settings = settings.clamped() if settings is not None else AdvancedTraceSettings()
    image_h, image_w = label_map.shape[:2]
    scaled_ratio_area = int(image_h * image_w * min_component_ratio)
    min_component_area = max(int(minimum_area_floor), scaled_ratio_area, int(settings.min_shape_area))

    layers: list[LayerResult] = []
    fallback_backend = False
    python_backend_used = False
    binary_backend_used = False
    raw_nodes_total = 0
    clean_nodes_total = 0
    regions_removed_total = 0
    pinholes_filled_total = 0

    for color in sorted(palette, key=lambda item: int(item["pixels"]), reverse=True):
        color_index = int(color["index"])
        raw_mask = np.where(label_map == color_index, 255, 0).astype(np.uint8)
        prepared_mask, cleanup_meta = _prepare_trace_mask(
            mask=raw_mask,
            min_component_area=min_component_area,
            settings=settings,
            quality_mode=quality_mode,
            preset=preset,
        )
        regions_removed_total += int(cleanup_meta["regionsRemoved"])
        pinholes_filled_total += int(cleanup_meta["pinholesFilled"])

        if cv2.countNonZero(prepared_mask) == 0:
            continue

        raw_paths: list[VectorPathData] = []
        potrace_settings = _resolve_potrace_settings(settings, quality_mode, preset)
        if py_potrace is not None:
            try:
                raw_paths = _trace_with_python_potrace(prepared_mask, potrace_settings)
                python_backend_used = len(raw_paths) > 0
            except Exception:  # pragma: no cover - runtime dependency
                raw_paths = []

        if not raw_paths and potrace_bin is not None and potrace_bin.exists():
            try:
                raw_paths = _trace_with_potrace_binary(prepared_mask, potrace_bin, potrace_settings)
                binary_backend_used = len(raw_paths) > 0
            except Exception:  # pragma: no cover - runtime dependency
                raw_paths = []

        if not raw_paths:
            raw_paths = _trace_with_contour_fallback(prepared_mask, settings)
            fallback_backend = True

        raw_nodes = sum(_estimate_node_count(item.d) for item in raw_paths)
        cleaned_paths = _cleanup_svg_paths(raw_paths)
        clean_nodes = sum(_estimate_node_count(item.d) for item in cleaned_paths)
        raw_nodes_total += raw_nodes
        clean_nodes_total += clean_nodes
        if not cleaned_paths:
            continue

        layers.append(
            LayerResult(
                index=color_index,
                hex_color=str(color["hex"]),
                pixel_count=int(color["pixels"]),
                mask=prepared_mask,
                svg_paths=cleaned_paths,
                node_count=clean_nodes,
            )
        )

    meta = {
        "colors_traced": len(layers),
        "min_component_area": int(min_component_area),
        "minimum_area_floor": int(minimum_area_floor),
        "used_backend": _backend_name(
            use_python=python_backend_used,
            use_binary=binary_backend_used,
            contour_fallback=fallback_backend,
        ),
        "cleanupStats": {
            "regionsRemoved": int(regions_removed_total),
            "pinholesFilled": int(pinholes_filled_total),
        },
        "nodeReductionStats": {
            "rawNodes": int(raw_nodes_total),
            "cleanNodes": int(clean_nodes_total),
            "removedNodes": int(max(0, raw_nodes_total - clean_nodes_total)),
        },
    }
    return layers, meta


def _backend_name(use_python: bool, use_binary: bool, contour_fallback: bool) -> str:
    if use_python:
        return "python-potrace"
    if use_binary:
        return "potrace-binary"
    if contour_fallback:
        return "opencv-contour-fallback"
    return "opencv-contour-fallback"


def _resolve_potrace_settings(
    settings: AdvancedTraceSettings,
    quality_mode: TraceQualityMode,
    preset: str,
) -> dict[str, float]:
    detail = settings.detail / 100.0
    smoothness = settings.smoothness / 100.0
    corners = settings.corners / 100.0
    despeckle = settings.despeckle / 100.0
    quality_bonus = 0.12 if quality_mode == "high_quality" else 0.0
    logo_bonus = 0.08 if preset == "logo" else 0.0
    text_bonus = 0.15 if preset == "text_typo" else 0.0
    dtf_bonus = 0.06 if preset == "dtf_screen_print" else 0.0
    sticker_bonus = 0.14 if preset == "sticker_cutting" else 0.0

    turdsize = max(
        1,
        int(round(1 + (1.0 - detail) * 4 + despeckle * 3 - quality_bonus * 4 - text_bonus * 4 + sticker_bonus * 3)),
    )
    alphamax = float(
        np.clip(
            1.15
            - corners * 0.85
            + smoothness * 0.25
            - logo_bonus
            - text_bonus * 0.75
            + dtf_bonus * 0.2
            + sticker_bonus * 0.35,
            0.0,
            1.35,
        )
    )
    opttolerance = float(
        np.clip(
            0.14 + smoothness * 0.34 - quality_bonus * 0.12 - text_bonus * 0.05 + dtf_bonus * 0.03 + sticker_bonus * 0.08,
            0.08,
            0.5,
        )
    )
    return {
        "turdsize": turdsize,
        "alphamax": round(alphamax, 2),
        "opttolerance": round(opttolerance, 2),
    }


def _prepare_trace_mask(
    mask: np.ndarray,
    min_component_area: int,
    settings: AdvancedTraceSettings,
    quality_mode: TraceQualityMode,
    preset: str,
) -> tuple[np.ndarray, dict[str, int]]:
    despeckle_ratio = settings.despeckle / 100.0
    detail_ratio = settings.detail / 100.0
    smooth_ratio = settings.smoothness / 100.0
    corners_ratio = settings.corners / 100.0

    removal_area = max(
        min_component_area,
        int(round(min_component_area * (0.8 + despeckle_ratio * 1.35 - detail_ratio * 0.35))),
    )
    opening_size = 1 if detail_ratio >= 0.65 else 3
    closing_size = 3 if smooth_ratio < 0.55 else 5
    if quality_mode == "high_quality" and closing_size < 5:
        closing_size += 2
    if corners_ratio >= 0.75:
        closing_size = max(3, closing_size - 2)
    if preset == "sticker_cutting":
        removal_area = max(removal_area, int(round(min_component_area * 1.45)))
        closing_size = max(closing_size, 5)
        opening_size = max(opening_size, 3)

    cleaned = _remove_small_regions(mask, removal_area)
    opened = cv2.morphologyEx(
        cleaned,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (opening_size, opening_size)),
        iterations=1,
    )
    closed = cv2.morphologyEx(
        opened,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (closing_size, closing_size)),
        iterations=1,
    )
    holes_area = max(4, removal_area // 2)
    if preset == "sticker_cutting":
        holes_area = max(holes_area, removal_area)
    filled, holes_count = _fill_small_holes(closed, holes_area)
    if quality_mode == "high_quality" and smooth_ratio < 0.75:
        filled = cv2.medianBlur(filled, 3)

    regions_before = _component_count(mask)
    regions_after = _component_count(filled)
    return filled, {
        "regionsRemoved": max(0, int(regions_before - regions_after)),
        "pinholesFilled": int(holes_count),
    }


def _trace_with_potrace_binary(mask: np.ndarray, potrace_bin: Path, settings: dict[str, float]) -> list[VectorPathData]:
    with tempfile.TemporaryDirectory(prefix="sablon_trace_") as tmp:
        tmpdir = Path(tmp)
        bitmap = tmpdir / "mask.bmp"
        traced_svg = tmpdir / "trace.svg"

        bitmap_img = np.full(mask.shape, 255, dtype=np.uint8)
        bitmap_img[mask > 0] = 0
        if not cv2.imwrite(str(bitmap), bitmap_img):
            raise OSError("Failed to write temporary bitmap for potrace.")

        command = [
            str(potrace_bin),
            str(bitmap),
            "-s",
            "-o",
            str(traced_svg),
            "-t",
            str(int(settings["turdsize"])),
            "-a",
            str(settings["alphamax"]),
            "-O",
            str(settings["opttolerance"]),
        ]
        subprocess.run(command, check=True, capture_output=True, text=True)
        return _read_svg_paths(traced_svg)


def _read_svg_paths(path: Path) -> list[VectorPathData]:
    root = ET.parse(path).getroot()
    group = root.find(f"{{{SVG_NS}}}g")
    transform = group.get("transform") if group is not None else None
    data: list[VectorPathData] = []

    candidates = group.findall(f"{{{SVG_NS}}}path") if group is not None else root.findall(f".//{{{SVG_NS}}}path")
    for item in candidates:
        command = item.get("d")
        if command:
            data.append(VectorPathData(d=command, transform=transform))
    return data


def _trace_with_python_potrace(mask: np.ndarray, settings: dict[str, float]) -> list[VectorPathData]:
    if py_potrace is None:
        return []

    bitmap = py_potrace.Bitmap((mask > 0).astype(np.uint32))
    traced = bitmap.trace(
        turdsize=int(settings["turdsize"]),
        alphamax=float(settings["alphamax"]),
        opticurve=True,
        opttolerance=float(settings["opttolerance"]),
    )
    paths: list[VectorPathData] = []

    for curve in traced:
        start = curve.start_point
        commands: list[str] = [f"M {start.x:.2f} {start.y:.2f}"]
        for segment in curve:
            if segment.is_corner:
                c = segment.c
                end = segment.end_point
                commands.append(f"L {c.x:.2f} {c.y:.2f} L {end.x:.2f} {end.y:.2f}")
            else:
                c1 = segment.c1
                c2 = segment.c2
                end = segment.end_point
                commands.append(
                    f"C {c1.x:.2f} {c1.y:.2f} {c2.x:.2f} {c2.y:.2f} {end.x:.2f} {end.y:.2f}"
                )
        commands.append("Z")
        paths.append(VectorPathData(d=" ".join(commands)))
    return paths


def _trace_with_contour_fallback(mask: np.ndarray, settings: AdvancedTraceSettings) -> list[VectorPathData]:
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_TC89_KCOS)
    paths: list[VectorPathData] = []
    epsilon_scale = 0.0016 + (settings.smoothness / 100.0) * 0.0045 - (settings.corners / 100.0) * 0.0012
    epsilon_scale = float(np.clip(epsilon_scale, 0.001, 0.006))

    del hierarchy
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < max(4, settings.min_shape_area // 2):
            continue
        reduced = _reduce_collinear_points(contour.reshape(-1, 2))
        contour_for_arc = reduced.reshape(-1, 1, 2)
        epsilon = max(1.0, epsilon_scale * cv2.arcLength(contour_for_arc, True))
        approx = cv2.approxPolyDP(contour_for_arc, epsilon, True)
        if len(approx) < 3:
            continue

        points = approx.reshape(-1, 2)
        commands = [f"M {points[0][0]:.2f} {points[0][1]:.2f}"]
        for point in points[1:]:
            commands.append(f"L {point[0]:.2f} {point[1]:.2f}")
        commands.append("Z")
        paths.append(VectorPathData(d=" ".join(commands)))
    return paths


def _remove_small_regions(region: np.ndarray, min_area: int) -> np.ndarray:
    components, labels, stats, _ = cv2.connectedComponentsWithStats(region, connectivity=8)
    cleaned = np.zeros_like(region)
    for idx in range(1, components):
        area = stats[idx, cv2.CC_STAT_AREA]
        if area >= min_area:
            cleaned[labels == idx] = 255
    return cleaned


def _fill_small_holes(mask: np.ndarray, max_hole_area: int) -> tuple[np.ndarray, int]:
    inverted = cv2.bitwise_not(mask)
    components, labels, stats, _ = cv2.connectedComponentsWithStats(inverted, connectivity=8)
    filled = mask.copy()
    holes_filled = 0
    height, width = mask.shape[:2]

    for component in range(1, components):
        x = int(stats[component, cv2.CC_STAT_LEFT])
        y = int(stats[component, cv2.CC_STAT_TOP])
        w = int(stats[component, cv2.CC_STAT_WIDTH])
        h = int(stats[component, cv2.CC_STAT_HEIGHT])
        area = int(stats[component, cv2.CC_STAT_AREA])
        touches_border = x == 0 or y == 0 or (x + w) >= width or (y + h) >= height
        if touches_border or area > max_hole_area:
            continue
        filled[labels == component] = 255
        holes_filled += 1

    return filled, holes_filled


def _component_count(mask: np.ndarray) -> int:
    if cv2.countNonZero(mask) == 0:
        return 0
    components, _, _, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    return max(0, int(components - 1))


def _reduce_collinear_points(points: np.ndarray) -> np.ndarray:
    if len(points) <= 3:
        return points

    reduced: list[np.ndarray] = [points[0]]
    for index in range(1, len(points) - 1):
        prev = reduced[-1]
        curr = points[index]
        nxt = points[index + 1]
        if _is_nearly_collinear(prev, curr, nxt):
            continue
        reduced.append(curr)
    reduced.append(points[-1])
    return np.asarray(reduced, dtype=np.int32)


def _is_nearly_collinear(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> bool:
    ax, ay = float(a[0]), float(a[1])
    bx, by = float(b[0]), float(b[1])
    cx, cy = float(c[0]), float(c[1])
    area = abs((bx - ax) * (cy - ay) - (by - ay) * (cx - ax))
    baseline = max(1.0, np.hypot(cx - ax, cy - ay))
    return (area / baseline) < 0.8


def _estimate_node_count(path_data: str) -> int:
    commands = re.findall(r"[MmLlHhVvCcSsQqTtAaZz]", path_data)
    return len(commands)


def _cleanup_svg_paths(paths: list[VectorPathData]) -> list[VectorPathData]:
    cleaned: list[VectorPathData] = []
    seen: set[str] = set()

    for item in paths:
        if not item.d:
            continue
        normalized = re.sub(r"\s+", " ", item.d).strip()
        normalized = re.sub(r"([A-Za-z])\s+\1", r"\1", normalized)
        if _estimate_node_count(normalized) < 2:
            continue
        key = f"{normalized}|{item.transform or ''}"
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(VectorPathData(d=normalized, transform=item.transform))
    return cleaned
