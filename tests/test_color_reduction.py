from __future__ import annotations

import numpy as np

from core.color_reduction import reduce_colors


def test_reduce_colors_respects_count() -> None:
    image = np.full((120, 180, 3), 255, dtype=np.uint8)
    image[:, :60] = (0, 0, 0)
    image[:, 60:120] = (0, 0, 255)
    image[:, 120:] = (0, 255, 0)

    quantized, label_map, palette, meta = reduce_colors(image, color_count=2)
    assert quantized.shape == image.shape
    assert label_map.shape == image.shape[:2]
    assert len(palette) == 2
    assert meta["actual_color_count"] == 2


def test_reduce_colors_removes_border_white_background_when_transparent() -> None:
    image = np.full((160, 220, 3), 255, dtype=np.uint8)
    image[30:130, 40:180] = (12, 120, 220)
    image[60:100, 80:140] = (255, 255, 255)

    quantized, label_map, palette, meta = reduce_colors(
        image,
        color_count=3,
        mask=np.full(image.shape[:2], 255, dtype=np.uint8),
        background_mode="transparent",
        ignore_white=False,
    )

    assert quantized.shape == image.shape
    assert np.all(label_map[0, :] == -1)
    assert np.all(label_map[:, 0] == -1)
    assert meta["background"]["removedComponents"] >= 1
    assert len(palette) >= 1
