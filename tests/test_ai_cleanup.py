from __future__ import annotations

import base64

import cv2
import numpy as np

from modules import ai_cleanup


def _encode_png(image: np.ndarray) -> str:
    success, encoded = cv2.imencode(".png", image)
    assert success
    return base64.b64encode(encoded.tobytes()).decode("utf-8")


def test_cleanup_image_builds_sd_webui_payload_and_restores_size(monkeypatch) -> None:
    payload_holder: dict[str, object] = {}
    response_image = np.full((1024, 683, 3), 255, dtype=np.uint8)
    cv2.rectangle(response_image, (120, 140), (560, 910), (30, 30, 30), thickness=12)

    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, object]:
            return {"images": [_encode_png(response_image)]}

    class FakeRequests:
        @staticmethod
        def post(url, json, timeout):
            payload_holder["url"] = url
            payload_holder["json"] = json
            payload_holder["timeout"] = timeout
            return FakeResponse()

    monkeypatch.setattr(ai_cleanup, "requests", FakeRequests)

    image = np.full((1200, 800, 3), 240, dtype=np.uint8)
    result = ai_cleanup.cleanup_image(
        image,
        base_url="http://127.0.0.1:7860",
        prompt="clean vector logo",
        negative_prompt="fabric texture",
        denoising_strength=0.35,
        steps=20,
        cfg_scale=6,
        connect_timeout_s=2.5,
        request_timeout_s=90.0,
    )

    payload = payload_holder["json"]
    assert payload_holder["url"] == "http://127.0.0.1:7860/sdapi/v1/img2img"
    assert payload["prompt"] == "clean vector logo"
    assert payload["negative_prompt"] == "fabric texture"
    assert payload["denoising_strength"] == 0.35
    assert payload["steps"] == 20
    assert payload["cfg_scale"] == 6.0
    assert payload["width"] == 683
    assert payload["height"] == 1024
    assert result.metadata["backend"] == "sd_webui"
    assert result.metadata["usedFallback"] is False
    assert result.cleaned_image.shape[:2] == image.shape[:2]


def test_cleanup_image_falls_back_when_sd_webui_is_offline(monkeypatch) -> None:
    class FailingRequests:
        @staticmethod
        def post(url, json, timeout):
            del url, json, timeout
            raise RuntimeError("connection refused")

    monkeypatch.setattr(ai_cleanup, "requests", FailingRequests)

    image = np.full((240, 240, 3), 200, dtype=np.uint8)
    result = ai_cleanup.cleanup_image(
        image,
        base_url="http://127.0.0.1:7860",
        prompt="clean vector logo",
        negative_prompt="fabric texture",
        denoising_strength=0.35,
        steps=20,
        cfg_scale=6,
        connect_timeout_s=2.5,
        request_timeout_s=90.0,
    )

    assert result.metadata["backend"] == "opencv_fallback"
    assert result.metadata["usedFallback"] is True
    assert "connection refused" in result.metadata["fallbackReason"]
    assert result.cleaned_image.shape == image.shape
