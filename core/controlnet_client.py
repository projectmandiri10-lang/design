from __future__ import annotations

import base64
import http.client
import json
import socket
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse

import cv2
import numpy as np

from .types import ControlNetPreprocessor, ControlNetSettings


CONTROLNET_PROMPT = (
    "clean black and white outline of the original t-shirt print, preserve composition, preserve typography, "
    "preserve logo shapes, crisp contours, vector-friendly line art"
)
CONTROLNET_NEGATIVE_PROMPT = (
    "new objects, extra text, distorted letters, altered composition, shading, gradients, texture, fabric, "
    "wrinkles, background clutter, blur, watercolor, photorealism"
)


@dataclass(slots=True)
class ControlNetDiscovery:
    version: int
    module: str
    model: str
    base_url: str


@dataclass(slots=True)
class ControlNetRenderResult:
    image: np.ndarray
    discovery: ControlNetDiscovery
    original_size: tuple[int, int]
    request_size: tuple[int, int]


class ControlNetClientError(RuntimeError):
    pass


class ControlNetClient:
    def __init__(self, settings: ControlNetSettings):
        self.settings = settings
        self.base_url = settings.normalized_base_url()
        self._connect_timeout = max(0.25, float(settings.connect_timeout_s))
        self._request_timeout = max(1.0, float(settings.request_timeout_s))
        parsed = urlparse(self.base_url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ControlNetClientError(f"Invalid A1111 URL: {self.base_url}")

        self._parsed = parsed
        self._base_path = parsed.path.rstrip("/")

    def discover(self, preprocessor: ControlNetPreprocessor) -> ControlNetDiscovery:
        version_data = self._request_json("GET", "/controlnet/version")
        version = int(version_data.get("version", 0))
        modules = self._extract_list(self._request_json("GET", "/controlnet/module_list"), ("module_list", "module_list_alias"))
        models = self._extract_list(self._request_json("GET", "/controlnet/model_list"), ("model_list",))

        if preprocessor not in modules:
            raise ControlNetClientError(f"ControlNet preprocessor '{preprocessor}' is not available in A1111.")

        model = self._resolve_model_name(models, self.settings.resolved_model_prefix())
        if model is None:
            raise ControlNetClientError(
                f"ControlNet model prefix '{self.settings.resolved_model_prefix()}' is not available in A1111."
            )

        return ControlNetDiscovery(
            version=version,
            module=preprocessor,
            model=model,
            base_url=self.base_url,
        )

    def render_outline(self, image: np.ndarray) -> ControlNetRenderResult:
        preprocessor = self.settings.validated_preprocessor()
        discovery = self.discover(preprocessor)
        request_image, original_size, request_size = _prepare_request_image(image, long_side_limit=1024)
        encoded_image = _encode_png_base64(request_image)

        payload = {
            "init_images": [encoded_image],
            "include_init_images": False,
            "prompt": CONTROLNET_PROMPT,
            "negative_prompt": CONTROLNET_NEGATIVE_PROMPT,
            "batch_size": 1,
            "n_iter": 1,
            "seed": 1,
            "denoising_strength": 0.2,
            "width": int(request_size[0]),
            "height": int(request_size[1]),
            "alwayson_scripts": {
                "controlnet": {
                    "args": [
                        {
                            "enabled": True,
                            "image": encoded_image,
                            "module": discovery.module,
                            "model": discovery.model,
                            "weight": 1.0,
                            "guidance_start": 0.0,
                            "guidance_end": 1.0,
                            "pixel_perfect": True,
                        }
                    ]
                }
            },
        }
        response = self._request_json("POST", "/sdapi/v1/img2img", payload)
        images = response.get("images")
        if not isinstance(images, list) or not images:
            raise ControlNetClientError("A1111 img2img response did not include generated images.")

        outline = _decode_base64_image(images[0])
        if outline.shape[1] != original_size[0] or outline.shape[0] != original_size[1]:
            outline = cv2.resize(outline, original_size, interpolation=cv2.INTER_CUBIC)

        return ControlNetRenderResult(
            image=outline,
            discovery=discovery,
            original_size=original_size,
            request_size=request_size,
        )

    def _request_json(self, method: str, endpoint: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        connection: http.client.HTTPConnection | http.client.HTTPSConnection | None = None
        body: str | None = None
        headers = {"Accept": "application/json"}
        if payload is not None:
            body = json.dumps(payload)
            headers["Content-Type"] = "application/json"

        path = f"{self._base_path}{endpoint}"
        if self._parsed.query:
            path = f"{path}?{self._parsed.query}"

        try:
            if self._parsed.scheme == "https":
                connection = http.client.HTTPSConnection(self._parsed.netloc, timeout=self._connect_timeout)
            else:
                connection = http.client.HTTPConnection(self._parsed.netloc, timeout=self._connect_timeout)

            connection.request(method=method, url=path, body=body, headers=headers)
            if connection.sock is not None:
                connection.sock.settimeout(self._request_timeout)
            response = connection.getresponse()
            raw = response.read()
        except socket.timeout as error:
            raise ControlNetClientError(f"A1111 request timed out: {error}.") from error
        except OSError as error:
            raise ControlNetClientError(f"Failed to connect to A1111 at {self.base_url}: {error}.") from error
        finally:
            if connection is not None:
                connection.close()

        if response.status >= 400:
            detail = raw.decode("utf-8", errors="replace").strip()
            raise ControlNetClientError(f"A1111 returned HTTP {response.status} for {endpoint}: {detail}")

        try:
            decoded = json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError as error:
            raise ControlNetClientError(f"A1111 returned invalid JSON for {endpoint}: {error}") from error

        if not isinstance(decoded, dict):
            raise ControlNetClientError(f"A1111 returned an unexpected payload for {endpoint}.")
        return decoded

    def _extract_list(self, payload: dict[str, Any], keys: tuple[str, ...]) -> list[str]:
        for key in keys:
            value = payload.get(key)
            if isinstance(value, list):
                return [str(item) for item in value]
        raise ControlNetClientError(f"A1111 response is missing one of the expected keys: {', '.join(keys)}")

    def _resolve_model_name(self, models: list[str], prefix: str) -> str | None:
        prefix_lower = prefix.lower()
        for candidate in models:
            if str(candidate).lower().startswith(prefix_lower):
                return str(candidate)
        return None


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
        raise ControlNetClientError("Failed to encode image for A1111 request.")
    return base64.b64encode(encoded.tobytes()).decode("utf-8")


def _decode_base64_image(value: str) -> np.ndarray:
    encoded = str(value)
    if encoded.startswith("data:") and "," in encoded:
        encoded = encoded.split(",", 1)[1]

    try:
        image_bytes = base64.b64decode(encoded)
    except Exception as error:  # pragma: no cover - base64 internals
        raise ControlNetClientError(f"Failed to decode A1111 image data: {error}") from error

    array = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(array, cv2.IMREAD_COLOR)
    if image is None:
        raise ControlNetClientError("A1111 returned image data that OpenCV could not decode.")
    return image
