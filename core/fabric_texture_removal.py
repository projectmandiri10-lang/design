from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

try:
    import onnxruntime as ort
except ImportError:  # pragma: no cover - runtime optional
    ort = None


@dataclass(slots=True)
class _OrtLayout:
    is_nchw: bool
    channels: int
    height: int
    width: int


class FabricDenoiserUNet:
    def __init__(self, model_path: Path):
        self.model_path = model_path
        self.session: ort.InferenceSession | None = None
        self.layout: _OrtLayout | None = None
        self.runtime_warning: str | None = None
        self.backend = "classical"
        self.model_kind = "generic"
        self.output_channels = 0

        if ort is None:
            self.runtime_warning = "onnxruntime is not installed. U-Net denoise uses classical fallback."
            return
        if not model_path.exists() or model_path.stat().st_size < 1024:
            self.runtime_warning = f"Model not ready: {model_path.name}. U-Net fallback mode enabled."
            return

        try:
            self.session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
            self.layout = _resolve_layout(self.session.get_inputs()[0].shape)
            self.model_kind = _infer_model_kind(model_path)
            self.output_channels = _infer_output_channels(self.session.get_outputs()[0].shape, self.layout)
            self.backend = "onnx"
        except Exception as error:  # pragma: no cover - runtime dependency
            self.runtime_warning = f"Failed to load U-Net ONNX: {error}. Fallback to classical denoise."
            self.session = None
            self.layout = None
            self.backend = "classical"

    @property
    def loaded(self) -> bool:
        return self.session is not None

    def remove(self, image: np.ndarray) -> tuple[np.ndarray, dict[str, object]]:
        if self.session is None or self.layout is None:
            return _classical_texture_removal(image), {"backend": self.backend, "fallback": True}

        try:
            tensor = _prepare_input(image, self.layout, model_kind=self.model_kind)
            input_name = self.session.get_inputs()[0].name
            output = self.session.run(None, {input_name: tensor})[0]

            if self.output_channels >= 2 or self.model_kind == "u2net":
                cloth_mask = _decode_segmentation_mask(output)
                cloth_mask = cv2.resize(cloth_mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)
                restored = _guided_texture_removal(image, cloth_mask)
                return restored, {
                    "backend": self.backend,
                    "fallback": False,
                    "cloth_coverage": round(float(np.mean(cloth_mask > 0.45)), 4),
                }

            restored = _decode_image_output(output)
            restored = cv2.resize(restored, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_CUBIC)
            return restored, {"backend": self.backend, "fallback": False}
        except Exception as error:  # pragma: no cover - runtime dependency
            self.runtime_warning = f"U-Net inference failed: {error}. Using classical fallback."
            return _classical_texture_removal(image), {"backend": "classical", "fallback": True}


def _classical_texture_removal(image: np.ndarray) -> np.ndarray:
    median = cv2.medianBlur(image, 3)
    bilateral = cv2.bilateralFilter(median, d=9, sigmaColor=55, sigmaSpace=55)

    gray = cv2.cvtColor(bilateral, cv2.COLOR_BGR2GRAY)
    open_texture = cv2.morphologyEx(
        gray,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        iterations=1,
    )
    detail = cv2.subtract(gray, open_texture)
    suppressed_gray = cv2.subtract(gray, cv2.GaussianBlur(detail, (3, 3), 0))
    suppressed_bgr = cv2.cvtColor(suppressed_gray, cv2.COLOR_GRAY2BGR)
    merged = cv2.addWeighted(bilateral, 0.7, suppressed_bgr, 0.3, 0)
    return merged


def _resolve_layout(shape: list[object]) -> _OrtLayout:
    if len(shape) != 4:
        return _OrtLayout(is_nchw=True, channels=3, height=512, width=512)

    dims = [value if isinstance(value, int) and value > 0 else None for value in shape]
    _, d1, d2, d3 = dims
    if d1 in (1, 3):
        return _OrtLayout(is_nchw=True, channels=d1, height=d2 or 768, width=d3 or 768)
    if d3 in (1, 3):
        return _OrtLayout(is_nchw=False, channels=d3, height=d1 or 768, width=d2 or 768)
    return _OrtLayout(is_nchw=True, channels=3, height=d2 or 768, width=d3 or 768)


def _prepare_input(image: np.ndarray, layout: _OrtLayout, model_kind: str) -> np.ndarray:
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (layout.width, layout.height), interpolation=cv2.INTER_LINEAR)
    if layout.channels == 1:
        resized = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY)[..., None]
        normalized = resized.astype(np.float32) / 255.0
    else:
        normalized = resized.astype(np.float32) / 255.0
        if model_kind == "u2net":
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
            normalized = (normalized - mean) / std
    if layout.is_nchw:
        return np.transpose(normalized, (2, 0, 1))[None, ...]
    return normalized[None, ...]


def _decode_image_output(output: np.ndarray) -> np.ndarray:
    array = np.asarray(output)
    if array.ndim == 4:
        if array.shape[1] in (1, 3):
            array = np.transpose(array[0], (1, 2, 0))
        elif array.shape[-1] in (1, 3):
            array = array[0]
        else:
            array = array[0, 0]
    elif array.ndim == 3:
        array = array[0] if array.shape[0] in (1, 3) else array
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


def _decode_segmentation_mask(output: np.ndarray) -> np.ndarray:
    array = np.asarray(output).astype(np.float32)
    if array.ndim == 4:
        if array.shape[1] <= 8:
            array = np.transpose(array[0], (1, 2, 0))
        elif array.shape[-1] <= 8:
            array = array[0]
        else:
            array = array[0, 0][..., None]
    elif array.ndim == 3:
        if array.shape[0] <= 8 and array.shape[2] > 8:
            array = np.transpose(array, (1, 2, 0))
        elif array.shape[-1] <= 8:
            array = array
        else:
            array = array[..., None]
    elif array.ndim == 2:
        array = array[..., None]
    else:
        return np.zeros((64, 64), dtype=np.float32)

    channels = array.shape[2]
    if channels == 1:
        mask = 1.0 / (1.0 + np.exp(-array[:, :, 0]))
    else:
        logits = array - np.max(array, axis=2, keepdims=True)
        exp = np.exp(logits)
        probs = exp / np.clip(np.sum(exp, axis=2, keepdims=True), 1e-6, None)
        if channels >= 4:
            background = probs[:, :, 0]
            mask = 1.0 - background
        else:
            mask = np.max(probs[:, :, 1:], axis=2)

    mask = np.clip(mask.astype(np.float32), 0.0, 1.0)
    mask = cv2.GaussianBlur(mask, (5, 5), 0)
    return mask


def _guided_texture_removal(image: np.ndarray, cloth_mask: np.ndarray) -> np.ndarray:
    classical = _classical_texture_removal(image).astype(np.float32)
    original = image.astype(np.float32)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 60, 150).astype(np.float32) / 255.0
    edge_soft = cv2.GaussianBlur(edges, (5, 5), 0)

    blend_mask = np.clip((cloth_mask - 0.20) / 0.80, 0.0, 1.0)
    blend_mask = np.clip(blend_mask * (1.0 - edge_soft), 0.0, 1.0)

    blend_3 = blend_mask[..., None]
    mixed = classical * blend_3 + original * (1.0 - blend_3)
    return np.clip(mixed, 0, 255).astype(np.uint8)


def _infer_model_kind(model_path: Path) -> str:
    if "u2net" in model_path.name.lower():
        return "u2net"
    return "generic"


def _infer_output_channels(shape: list[object], layout: _OrtLayout) -> int:
    if len(shape) != 4:
        return 0
    dims = [value if isinstance(value, int) and value > 0 else None for value in shape]
    _, d1, d2, d3 = dims
    if layout.is_nchw and d1 is not None:
        return int(d1)
    if (not layout.is_nchw) and d3 is not None:
        return int(d3)
    if d1 is not None and d1 <= 8:
        return int(d1)
    if d3 is not None and d3 <= 8:
        return int(d3)
    return 0
