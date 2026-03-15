from __future__ import annotations

import numpy as np

from core.vectorize_bitmap import vectorize_by_color_layers


def test_vectorize_layer_with_fallback_contour() -> None:
    label_map = np.full((120, 180), -1, dtype=np.int16)
    label_map[20:100, 30:150] = 0
    palette = [{"index": 0, "hex": "#000000", "pixels": int((80 * 120))}]

    layers, meta = vectorize_by_color_layers(
        label_map=label_map,
        palette=palette,
        min_component_ratio=0.0001,
        potrace_bin=None,
    )

    assert len(layers) == 1
    assert layers[0].node_count > 0
    assert meta["colors_traced"] == 1
