from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

try:
    import onnxruntime as ort
except ImportError:  # pragma: no cover - runtime optional
    ort = None

from .types import DetectionResult


class SablonAreaDetector:
    def __init__(self, model_path: Path):
        self.model_path = model_path
        self.session: ort.InferenceSession | None = None
        self.input_name: str | None = None
        self.runtime_warning: str | None = None
        self.backend = "contour"
        self.model_kind = "generic"

        if ort is None:
            self.runtime_warning = "onnxruntime is unavailable. Detector uses contour fallback."
            return
        if not model_path.exists() or model_path.stat().st_size < 1024:
            self.runtime_warning = f"Detector model not ready: {model_path.name}. Using contour fallback."
            return

        try:
            self.session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
            self.input_name = self.session.get_inputs()[0].name
            self.model_kind = _infer_model_kind(model_path, self.session)
            self.backend = "onnx"
        except Exception as error:  # pragma: no cover - runtime dependency
            self.runtime_warning = f"Detector ONNX load failed: {error}. Using contour fallback."
            self.session = None
            self.input_name = None
            self.backend = "contour"

    @property
    def loaded(self) -> bool:
        return self.session is not None

    def detect(self, image: np.ndarray) -> DetectionResult:
        if self.session is not None:
            onnx_result = self._detect_with_onnx(image)
            if onnx_result is not None:
                return onnx_result

        return self._detect_with_contour(image)

    def _detect_with_onnx(self, image: np.ndarray) -> DetectionResult | None:
        assert self.session is not None
        assert self.input_name is not None

        try:
            input_meta = self.session.get_inputs()[0]
            tensor, output_size = _prepare_input_for_model(
                image=image,
                shape=input_meta.shape,
                model_kind=self.model_kind,
            )
            outputs = self.session.run(None, {self.input_name: tensor})
            output = outputs[0]
            mask = _output_to_mask(output=output, expected_size=output_size, model_kind=self.model_kind)
            mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)
            coverage = float(np.count_nonzero(mask)) / float(mask.size)
            if coverage < 0.005 or coverage > 0.98:
                return None
            return _extract_roi(image, mask, method="onnx")
        except Exception as error:  # pragma: no cover - runtime dependency
            self.runtime_warning = f"Detector inference failed: {error}. Falling back to contour."
            return None

    def _detect_with_contour(self, image: np.ndarray) -> DetectionResult:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        adaptive = cv2.adaptiveThreshold(
            blur,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            35,
            4,
        )
        _, otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        mask = cv2.bitwise_or(adaptive, otsu)
        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
            iterations=2,
        )
        mask = cv2.medianBlur(mask, 5)
        return _extract_roi(image, mask, method="contour")


def _prepare_input_for_model(
    image: np.ndarray,
    shape: list[object],
    model_kind: str,
) -> tuple[np.ndarray, tuple[int, int]]:
    default_size = 640
    target_h, target_w = default_size, default_size
    is_nchw = True
    channels = 3

    if len(shape) == 4:
        dim = [value if isinstance(value, int) and value > 0 else None for value in shape]
        _, d1, d2, d3 = dim
        if d1 in (1, 3):
            is_nchw = True
            channels = d1
            target_h, target_w = d2 or default_size, d3 or default_size
        elif d3 in (1, 3):
            is_nchw = False
            channels = d3
            target_h, target_w = d1 or default_size, d2 or default_size

    resized = cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

    if channels == 1:
        rgb = cv2.cvtColor((rgb * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
        rgb = rgb[..., None]
    elif model_kind == "u2net":
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        rgb = (rgb - mean) / std

    if is_nchw:
        tensor = np.transpose(rgb, (2, 0, 1))[None, ...]
    else:
        tensor = rgb[None, ...]
    return tensor.astype(np.float32), (target_h, target_w)


def _output_to_mask(output: np.ndarray, expected_size: tuple[int, int], model_kind: str) -> np.ndarray:
    result = np.asarray(output)

    if result.ndim == 4:
        if result.shape[1] in (1, 2, 3):
            result = result[0, 0]
        elif result.shape[-1] in (1, 2, 3):
            result = result[0, :, :, 0]
        else:
            result = result[0, 0]
    elif result.ndim == 3:
        result = result[0]
    elif result.ndim == 2:
        result = result
    else:
        result = np.zeros(expected_size, dtype=np.float32)

    result = result.astype(np.float32)
    if float(result.min()) < 0.0 or float(result.max()) > 1.0:
        result = 1.0 / (1.0 + np.exp(-result))

    result -= result.min()
    max_value = float(result.max())
    if max_value > 0:
        result /= max_value

    threshold = 0.40 if model_kind == "u2net" else 0.45
    mask = np.where(result > threshold, 255, 0).astype(np.uint8)
    mask = cv2.medianBlur(mask, 5)
    mask = _keep_largest_component(mask)
    return cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
        iterations=1,
    )


def _keep_largest_component(mask: np.ndarray) -> np.ndarray:
    components, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if components <= 1:
        return mask

    areas = stats[1:, cv2.CC_STAT_AREA]
    index = int(np.argmax(areas)) + 1
    cleaned = np.zeros_like(mask)
    cleaned[labels == index] = 255
    return cleaned


def _infer_model_kind(model_path: Path, session: ort.InferenceSession) -> str:
    name = model_path.name.lower()
    if "u2net" in name:
        return "u2net"

    try:
        outputs = session.get_outputs()
        if len(outputs) >= 2 and all(
            isinstance(output.shape, list) and len(output.shape) == 4 for output in outputs[:2]
        ):
            return "u2net"
    except Exception:
        pass
    return "generic"


def _extract_roi(image: np.ndarray, mask: np.ndarray, method: str) -> DetectionResult:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        fallback_mask = np.full(image.shape[:2], 255, dtype=np.uint8)
        return DetectionResult(
            cropped_image=image.copy(),
            bbox_xywh=(0, 0, image.shape[1], image.shape[0]),
            confidence=0.0,
            method=f"{method}_fallback_full",
            mask=fallback_mask,
        )

    image_area = float(image.shape[0] * image.shape[1])
    contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(contour)
    area = float(cv2.contourArea(contour))

    pad_x = max(4, int(w * 0.08))
    pad_y = max(4, int(h * 0.08))
    x0 = max(0, x - pad_x)
    y0 = max(0, y - pad_y)
    x1 = min(image.shape[1], x + w + pad_x)
    y1 = min(image.shape[0], y + h + pad_y)

    roi = image[y0:y1, x0:x1].copy()
    roi_mask = mask[y0:y1, x0:x1].copy()
    confidence = min(1.0, max(0.0, area / max(1.0, image_area)))

    if (x1 - x0) < 48 or (y1 - y0) < 48:
        roi = image.copy()
        roi_mask = np.full(image.shape[:2], 255, dtype=np.uint8)
        x0, y0, x1, y1 = 0, 0, image.shape[1], image.shape[0]
        confidence = 0.05
        method = f"{method}_small_roi_full"

    return DetectionResult(
        cropped_image=roi,
        bbox_xywh=(x0, y0, x1 - x0, y1 - y0),
        confidence=round(confidence, 4),
        method=method,
        mask=roi_mask,
    )
