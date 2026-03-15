from __future__ import annotations

import base64
from dataclasses import dataclass

import cv2
import numpy as np

try:
    import requests
except ImportError:  # pragma: no cover - runtime optional
    requests = None


@dataclass(slots=True)
class CleanupArtifacts:
    cleaned_image: np.ndarray
    metadata: dict[str, object]


def cleanup_image(
    image: np.ndarray,
    *,
    base_url: str,
    prompt: str,
    negative_prompt: str,
    denoising_strength: float,
    steps: int,
    cfg_scale: float,
    connect_timeout_s: float,
    request_timeout_s: float,
) -> CleanupArtifacts:
    if requests is None:
        fallback = fallback_cleanup_image(image)
        return CleanupArtifacts(
            cleaned_image=fallback,
            metadata={
                "backend": "opencv_fallback",
                "usedFallback": True,
                "fallbackReason": "requests is not installed.",
            },
        )

    resized, original_size, request_size = _prepare_request_image(image, long_side_limit=1024)
    encoded = _encode_png_base64(resized)
    payload = {
        "init_images": [encoded],
        "include_init_images": False,
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "denoising_strength": float(denoising_strength),
        "steps": int(steps),
        "cfg_scale": float(cfg_scale),
        "batch_size": 1,
        "n_iter": 1,
        "width": int(request_size[0]),
        "height": int(request_size[1]),
    }
    endpoint = f"{base_url.rstrip('/')}/sdapi/v1/img2img"

    try:
        response = requests.post(
            endpoint,
            json=payload,
            timeout=(float(connect_timeout_s), float(request_timeout_s)),
        )
        response.raise_for_status()
        body = response.json()
        images = body.get("images")
        if not isinstance(images, list) or not images:
            raise RuntimeError("Stable Diffusion img2img response did not include images.")
        cleaned = _decode_base64_image(images[0])
        if cleaned.shape[1] != original_size[0] or cleaned.shape[0] != original_size[1]:
            cleaned = cv2.resize(cleaned, original_size, interpolation=cv2.INTER_CUBIC)
        return CleanupArtifacts(
            cleaned_image=cleaned,
            metadata={
                "backend": "sd_webui",
                "usedFallback": False,
                "baseUrl": base_url.rstrip("/"),
                "requestSize": {"width": int(request_size[0]), "height": int(request_size[1])},
                "prompt": prompt,
                "negativePrompt": negative_prompt,
                "denoisingStrength": float(denoising_strength),
                "steps": int(steps),
                "cfgScale": float(cfg_scale),
            },
        )
    except Exception as error:
        fallback = fallback_cleanup_image(image)
        return CleanupArtifacts(
            cleaned_image=fallback,
            metadata={
                "backend": "opencv_fallback",
                "usedFallback": True,
                "fallbackReason": str(error),
                "baseUrl": base_url.rstrip("/"),
            },
        )


def fallback_cleanup_image(image: np.ndarray) -> np.ndarray:
    denoised = cv2.fastNlMeansDenoisingColored(image, None, 5, 5, 7, 21)
    bilateral = cv2.bilateralFilter(denoised, d=9, sigmaColor=60, sigmaSpace=60)
    sharpen = cv2.addWeighted(bilateral, 1.12, cv2.GaussianBlur(bilateral, (0, 0), 1.2), -0.12, 0)
    return sharpen


def _prepare_request_image(image: np.ndarray, long_side_limit: int) -> tuple[np.ndarray, tuple[int, int], tuple[int, int]]:
    original_h, original_w = image.shape[:2]
    original_size = (int(original_w), int(original_h))
    max_side = max(original_h, original_w)
    if max_side <= long_side_limit:
        return image, original_size, original_size

    scale = float(long_side_limit) / float(max_side)
    target_w = max(1, int(round(original_w * scale)))
    target_h = max(1, int(round(original_h * scale)))
    resized = cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_AREA)
    return resized, original_size, (target_w, target_h)


def _encode_png_base64(image: np.ndarray) -> str:
    success, encoded = cv2.imencode(".png", image)
    if not success:
        raise RuntimeError("Failed to encode image for Stable Diffusion request.")
    return base64.b64encode(encoded.tobytes()).decode("utf-8")


def _decode_base64_image(value: str) -> np.ndarray:
    encoded = str(value)
    if encoded.startswith("data:") and "," in encoded:
        encoded = encoded.split(",", 1)[1]

    image_bytes = base64.b64decode(encoded)
    array = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(array, cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError("Stable Diffusion returned image data that OpenCV could not decode.")
    return image
