from __future__ import annotations

import cv2
import numpy as np

try:
    from sklearn.cluster import KMeans
except ImportError:  # pragma: no cover - runtime optional
    KMeans = None

from .types import BackgroundMode


def _bgr_to_hex(color: np.ndarray) -> str:
    blue, green, red = [int(round(float(channel))) for channel in color]
    return f"#{red:02x}{green:02x}{blue:02x}"


def _ensure_foreground_mask(image: np.ndarray, mask: np.ndarray | None) -> np.ndarray:
    if mask is not None and np.count_nonzero(mask) > 0:
        return (mask > 0).astype(np.uint8) * 255

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    fallback = np.where(gray < 250, 255, 0).astype(np.uint8)
    return fallback


def reduce_colors(
    image: np.ndarray,
    color_count: int,
    mask: np.ndarray | None = None,
    random_seed: int = 42,
    min_region_area: int = 18,
    background_mode: BackgroundMode = "transparent",
    ignore_white: bool = False,
    quality_mode: str = "balanced",
) -> tuple[np.ndarray, np.ndarray, list[dict[str, object]], dict[str, object]]:
    foreground_mask = _ensure_foreground_mask(image, mask)
    pixels = image[foreground_mask > 0].reshape(-1, 3)
    if pixels.size == 0:
        raise ValueError("No printable foreground after masking.")

    unique_count = len(np.unique(pixels, axis=0))
    clusters = max(1, min(int(color_count), unique_count))
    sample_cap = 80000 if quality_mode == "high_quality" else 50000

    if len(pixels) > sample_cap:
        rng = np.random.default_rng(random_seed)
        sample_idx = rng.choice(len(pixels), size=sample_cap, replace=False)
        fit_pixels = pixels[sample_idx]
    else:
        fit_pixels = pixels

    labels, centers = _cluster_pixels(
        fit_pixels=fit_pixels.astype(np.float32),
        full_pixels=pixels.astype(np.float32),
        clusters=clusters,
        random_seed=random_seed,
    )

    raw_counts = np.bincount(labels, minlength=clusters)
    luminance = centers @ np.array([0.114, 0.587, 0.299])
    order = sorted(range(clusters), key=lambda idx: (-raw_counts[idx], luminance[idx]))

    remap = np.zeros(clusters, dtype=np.uint8)
    for new_idx, old_idx in enumerate(order):
        remap[old_idx] = new_idx
    remapped_labels = remap[labels]
    ordered_centers = centers[order]

    label_map = np.full(foreground_mask.shape, -1, dtype=np.int16)
    label_map[foreground_mask > 0] = remapped_labels.astype(np.int16)

    label_map, cleanup_meta = _clean_label_regions(
        label_map=label_map,
        cluster_count=clusters,
        min_region_area=max(1, int(min_region_area)),
        quality_mode=quality_mode,
    )
    label_map, white_meta = _apply_background_mode(
        label_map=label_map,
        centers=ordered_centers,
        background_mode=background_mode,
        ignore_white=ignore_white,
    )
    label_map, ordered_centers = _reindex_active_labels(label_map, ordered_centers)

    fg = label_map >= 0
    if not np.any(fg):
        raise ValueError("Color reduction removed all printable layers.")

    quantized = np.full_like(image, 255)
    quantized[fg] = ordered_centers[label_map[fg]]

    pixel_counts = np.bincount(label_map[fg], minlength=len(ordered_centers))
    palette = [
        {
            "index": int(index),
            "hex": _bgr_to_hex(color),
            "rgb": [int(color[2]), int(color[1]), int(color[0])],
            "pixels": int(pixel_counts[index]),
        }
        for index, color in enumerate(ordered_centers)
        if int(pixel_counts[index]) > 0
    ]

    meta = {
        "requested_color_count": int(color_count),
        "actual_color_count": int(len(palette)),
        "backend": "sklearn" if KMeans is not None else "opencv-kmeans-fallback",
        "cleanup": cleanup_meta,
        "background": white_meta,
    }
    return quantized, label_map, palette, meta


def _cluster_pixels(
    fit_pixels: np.ndarray,
    full_pixels: np.ndarray,
    clusters: int,
    random_seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    if KMeans is not None:
        kmeans = KMeans(n_clusters=clusters, n_init=8, random_state=random_seed)
        kmeans.fit(fit_pixels)
        labels = kmeans.predict(full_pixels)
        centers = np.clip(kmeans.cluster_centers_, 0, 255).astype(np.uint8)
        return labels.astype(np.int32), centers

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 24, 0.3)
    _compactness, fit_labels, centers = cv2.kmeans(
        fit_pixels,
        clusters,
        None,
        criteria,
        4,
        cv2.KMEANS_PP_CENTERS,
    )
    fit_labels = fit_labels.reshape(-1)

    if len(full_pixels) == len(fit_pixels):
        labels = fit_labels
    else:
        labels = _nearest_center_labels(full_pixels, centers)

    centers = np.clip(centers, 0, 255).astype(np.uint8)
    return labels.astype(np.int32), centers


def _nearest_center_labels(pixels: np.ndarray, centers: np.ndarray) -> np.ndarray:
    distances = np.linalg.norm(pixels[:, None, :] - centers[None, :, :], axis=2)
    return np.argmin(distances, axis=1).astype(np.int32)


def _clean_label_regions(
    label_map: np.ndarray,
    cluster_count: int,
    min_region_area: int,
    quality_mode: str,
) -> tuple[np.ndarray, dict[str, object]]:
    cleaned = label_map.copy()
    regions_removed = 0
    pinholes_filled = 0
    closing_kernel = 3 if quality_mode == "balanced" else 5
    shape = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (closing_kernel, closing_kernel))

    for index in range(cluster_count):
        mask = np.where(cleaned == index, 255, 0).astype(np.uint8)
        if cv2.countNonZero(mask) == 0:
            continue

        components, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        layer_mask = np.zeros_like(mask)
        for component in range(1, components):
            area = int(stats[component, cv2.CC_STAT_AREA])
            if area >= min_region_area:
                layer_mask[labels == component] = 255
            else:
                regions_removed += 1

        filled_mask, filled_count = _fill_small_holes(layer_mask, max(4, min_region_area // 2))
        pinholes_filled += filled_count
        layer_mask = cv2.morphologyEx(filled_mask, cv2.MORPH_CLOSE, shape, iterations=1)
        cleaned[mask > 0] = -1
        cleaned[layer_mask > 0] = index

    smoothed = np.zeros_like(cleaned, dtype=np.uint8)
    active = cleaned >= 0
    smoothed[active] = cleaned[active].astype(np.uint8) + 1
    smoothed = cv2.medianBlur(smoothed, 3 if quality_mode == "balanced" else 5)
    cleaned = np.where(smoothed > 0, smoothed.astype(np.int16) - 1, -1)

    return cleaned, {
        "regionsRemoved": int(regions_removed),
        "pinholesFilled": int(pinholes_filled),
        "minRegionArea": int(min_region_area),
    }


def _fill_small_holes(mask: np.ndarray, max_hole_area: int) -> tuple[np.ndarray, int]:
    if cv2.countNonZero(mask) == 0:
        return mask, 0

    inverted = cv2.bitwise_not(mask)
    components, labels, stats, _ = cv2.connectedComponentsWithStats(inverted, connectivity=8)
    filled = mask.copy()
    count = 0
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
        count += 1
    return filled, count


def _apply_background_mode(
    label_map: np.ndarray,
    centers: np.ndarray,
    background_mode: BackgroundMode,
    ignore_white: bool,
    white_threshold: int = 245,
) -> tuple[np.ndarray, dict[str, object]]:
    cleaned = label_map.copy()
    removed_components = 0
    removed_labels = 0
    near_white = np.all(centers >= white_threshold, axis=1)

    if not np.any(near_white):
        return cleaned, {
            "mode": background_mode,
            "ignoreWhite": bool(ignore_white),
            "removedComponents": 0,
            "removedLabels": 0,
        }

    for index, is_white in enumerate(near_white):
        if not is_white:
            continue
        layer_mask = np.where(cleaned == index, 255, 0).astype(np.uint8)
        if cv2.countNonZero(layer_mask) == 0:
            continue

        if ignore_white:
            removed_components += _remove_all_components(cleaned, index)
            removed_labels += 1
            continue

        if background_mode in {"transparent", "drop_white"}:
            removed_components += _remove_border_connected_components(cleaned, index)
            removed_labels += 1

    return cleaned, {
        "mode": background_mode,
        "ignoreWhite": bool(ignore_white),
        "removedComponents": int(removed_components),
        "removedLabels": int(removed_labels),
    }


def _remove_all_components(label_map: np.ndarray, index: int) -> int:
    removed = int(np.count_nonzero(label_map == index) > 0)
    label_map[label_map == index] = -1
    return removed


def _remove_border_connected_components(label_map: np.ndarray, index: int) -> int:
    mask = np.where(label_map == index, 255, 0).astype(np.uint8)
    components, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    removed = 0
    height, width = mask.shape[:2]

    for component in range(1, components):
        x = int(stats[component, cv2.CC_STAT_LEFT])
        y = int(stats[component, cv2.CC_STAT_TOP])
        w = int(stats[component, cv2.CC_STAT_WIDTH])
        h = int(stats[component, cv2.CC_STAT_HEIGHT])
        touches_border = x == 0 or y == 0 or (x + w) >= width or (y + h) >= height
        if not touches_border:
            continue
        label_map[labels == component] = -1
        removed += 1
    return removed


def _reindex_active_labels(label_map: np.ndarray, centers: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    active = sorted(int(index) for index in np.unique(label_map) if index >= 0)
    if not active:
        return label_map, centers[:0]

    remapped = np.full_like(label_map, -1)
    new_centers: list[np.ndarray] = []
    for new_index, old_index in enumerate(active):
        remapped[label_map == old_index] = new_index
        new_centers.append(centers[old_index])
    return remapped, np.asarray(new_centers, dtype=np.uint8)
