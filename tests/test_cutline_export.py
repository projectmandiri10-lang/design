from __future__ import annotations

import numpy as np

from core.cutline import build_cutline_paths
from core.types import LayerResult, VectorPathData
from export.export_svg import compose_cutline_svg_string


def test_build_cutline_paths_creates_external_offset_contour() -> None:
    mask = np.zeros((140, 180), dtype=np.uint8)
    mask[30:110, 40:140] = 255
    layer = LayerResult(
        index=0,
        hex_color="#000000",
        pixel_count=int(np.count_nonzero(mask)),
        mask=mask,
        svg_paths=[VectorPathData(d="M 40 30 L 140 30 L 140 110 L 40 110 Z")],
        node_count=5,
    )

    cutline_paths, cutline_mask, meta = build_cutline_paths([layer], offset_px=10)

    assert len(cutline_paths) == 1
    assert meta["offsetPx"] == 10
    assert meta["pathCount"] == 1
    assert cutline_mask.shape == mask.shape
    assert int(np.count_nonzero(cutline_mask)) > int(np.count_nonzero(mask))


def test_compose_cutline_svg_string_outputs_stroke_only_group() -> None:
    svg = compose_cutline_svg_string(
        canvas_size=(200, 100),
        cutline_paths=[VectorPathData(d="M 10 10 L 190 10 L 190 90 L 10 90 Z")],
        title="Cutline Test",
    )

    assert svg.startswith("<svg")
    assert 'id="cutline"' in svg
    assert 'fill="none"' in svg
    assert 'stroke="#ff00ff"' in svg
