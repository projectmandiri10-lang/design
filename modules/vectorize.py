from __future__ import annotations

from dataclasses import dataclass

from core.types import LayerResult, PipelineConfig
from core.vectorize_bitmap import detect_potrace_binary, vectorize_by_color_layers
from export.export_svg import compose_svg_string


@dataclass(slots=True)
class VectorArtifacts:
    layers: list[LayerResult]
    vector_svg: str
    metadata: dict[str, object]


def generate_vector_svg(
    *,
    label_map,
    palette: list[dict[str, object]],
    canvas_size: tuple[int, int],
    config: PipelineConfig,
    title: str,
) -> VectorArtifacts:
    potrace_bin = detect_potrace_binary(config.paths.potrace_bin)
    layers, vector_meta = vectorize_by_color_layers(
        label_map=label_map,
        palette=palette,
        min_component_ratio=config.min_component_ratio_photo,
        potrace_bin=potrace_bin,
        minimum_area_floor=config.settings.clamped().min_shape_area,
        settings=config.settings,
        quality_mode=config.validated_quality_mode(),
        preset="photo",
    )
    if not layers:
        raise RuntimeError("Vectorization produced no layers.")

    svg_text = compose_svg_string(
        canvas_size=canvas_size,
        layers=layers,
        title=title,
        background_mode=config.validated_background_mode(),
    )
    return VectorArtifacts(
        layers=layers,
        vector_svg=svg_text,
        metadata=vector_meta,
    )
