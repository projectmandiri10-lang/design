from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from core.color_reduction import reduce_colors
from core.types import PipelineConfig


@dataclass(slots=True)
class ColorReductionArtifacts:
    quantized: np.ndarray
    label_map: np.ndarray
    palette: list[dict[str, object]]
    metadata: dict[str, object]


def reduce_palette(
    image: np.ndarray,
    *,
    foreground_mask: np.ndarray | None,
    color_count: int,
    config: PipelineConfig,
) -> ColorReductionArtifacts:
    quantized, label_map, palette, metadata = reduce_colors(
        image=image,
        color_count=color_count,
        mask=foreground_mask,
        min_region_area=config.settings.clamped().min_shape_area,
        background_mode=config.validated_background_mode(),
        ignore_white=config.settings.clamped().ignore_white,
        quality_mode=config.validated_quality_mode(),
    )
    return ColorReductionArtifacts(
        quantized=quantized,
        label_map=label_map,
        palette=palette,
        metadata=metadata,
    )
