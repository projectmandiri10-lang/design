from __future__ import annotations

import base64
import socket
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from core.controlnet_client import ControlNetClient, ControlNetClientError
from core.types import ControlNetSettings


def _encode_png(image: np.ndarray) -> str:
    success, encoded = cv2.imencode(".png", image)
    assert success
    return base64.b64encode(encoded.tobytes()).decode("utf-8")


def test_controlnet_client_render_outline_resolves_model_prefix_and_builds_payload() -> None:
    settings = ControlNetSettings(enabled=True, preprocessor="lineart")
    client = ControlNetClient(settings)
    recorded: dict[str, object] = {}

    request_image = np.full((1200, 800, 3), 240, dtype=np.uint8)
    cv2.rectangle(request_image, (180, 160), (620, 980), (40, 40, 40), thickness=16)

    response_image = np.full((1024, 683, 3), 255, dtype=np.uint8)
    cv2.rectangle(response_image, (130, 120), (552, 912), (0, 0, 0), thickness=10)

    def fake_request_json(method: str, endpoint: str, payload=None):
        recorded.setdefault("calls", []).append((method, endpoint))
        if endpoint == "/controlnet/version":
            return {"version": 1}
        if endpoint == "/controlnet/module_list":
            return {"module_list": ["canny", "lineart"]}
        if endpoint == "/controlnet/model_list":
            return {"model_list": ["control_v11p_sd15_lineart [abcd1234]"]}
        if endpoint == "/sdapi/v1/img2img":
            recorded["payload"] = payload
            return {"images": [_encode_png(response_image)]}
        raise AssertionError(endpoint)

    client._request_json = fake_request_json  # type: ignore[method-assign]
    result = client.render_outline(request_image)

    payload = recorded["payload"]
    assert isinstance(payload, dict)
    assert payload["width"] == 683
    assert payload["height"] == 1024
    assert payload["batch_size"] == 1
    assert payload["denoising_strength"] == 0.2

    unit = payload["alwayson_scripts"]["controlnet"]["args"][0]
    assert unit["module"] == "lineart"
    assert unit["model"] == "control_v11p_sd15_lineart [abcd1234]"
    assert unit["weight"] == 1.0
    assert unit["guidance_start"] == 0.0
    assert unit["guidance_end"] == 1.0
    assert unit["pixel_perfect"] is True

    assert result.discovery.model == "control_v11p_sd15_lineart [abcd1234]"
    assert result.request_size == (683, 1024)
    assert result.image.shape[:2] == request_image.shape[:2]


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        ({"module_list": ["canny"]}, "preprocessor 'lineart'"),
        ({"model_list": ["control_v11p_sd15_canny [hash]"]}, "model prefix 'control_v11p_sd15_lineart'"),
    ],
)
def test_controlnet_client_discover_requires_available_module_and_model(payload: dict[str, object], expected: str) -> None:
    settings = ControlNetSettings(enabled=True, preprocessor="lineart")
    client = ControlNetClient(settings)

    def fake_request_json(method: str, endpoint: str, body=None):
        del method, body
        if endpoint == "/controlnet/version":
            return {"version": 1}
        if endpoint == "/controlnet/module_list":
            return payload if "module_list" in payload else {"module_list": ["lineart"]}
        if endpoint == "/controlnet/model_list":
            return payload if "model_list" in payload else {"model_list": ["control_v11p_sd15_lineart [hash]"]}
        raise AssertionError(endpoint)

    client._request_json = fake_request_json  # type: ignore[method-assign]

    with pytest.raises(ControlNetClientError, match=expected):
        client.discover("lineart")


def test_controlnet_client_request_timeout_surfaces_structured_error(monkeypatch: pytest.MonkeyPatch) -> None:
    class TimeoutConnection:
        def __init__(self, *args, **kwargs):
            self.sock = SimpleNamespace(settimeout=lambda value: None)

        def request(self, *args, **kwargs):
            raise socket.timeout("boom")

        def close(self):
            return None

    monkeypatch.setattr("core.controlnet_client.http.client.HTTPConnection", TimeoutConnection)
    client = ControlNetClient(ControlNetSettings(enabled=True))

    with pytest.raises(ControlNetClientError, match="timed out"):
        client._request_json("GET", "/controlnet/version")


def test_controlnet_client_request_http_error(monkeypatch: pytest.MonkeyPatch) -> None:
    class ErrorResponse:
        status = 500

        def read(self):
            return b"internal error"

    class ErrorConnection:
        def __init__(self, *args, **kwargs):
            self.sock = SimpleNamespace(settimeout=lambda value: None)

        def request(self, *args, **kwargs):
            return None

        def getresponse(self):
            return ErrorResponse()

        def close(self):
            return None

    monkeypatch.setattr("core.controlnet_client.http.client.HTTPConnection", ErrorConnection)
    client = ControlNetClient(ControlNetSettings(enabled=True))

    with pytest.raises(ControlNetClientError, match="HTTP 500"):
        client._request_json("GET", "/controlnet/version")


def test_controlnet_client_request_invalid_json(monkeypatch: pytest.MonkeyPatch) -> None:
    class InvalidJsonResponse:
        status = 200

        def read(self):
            return b"{invalid"

    class InvalidJsonConnection:
        def __init__(self, *args, **kwargs):
            self.sock = SimpleNamespace(settimeout=lambda value: None)

        def request(self, *args, **kwargs):
            return None

        def getresponse(self):
            return InvalidJsonResponse()

        def close(self):
            return None

    monkeypatch.setattr("core.controlnet_client.http.client.HTTPConnection", InvalidJsonConnection)
    client = ControlNetClient(ControlNetSettings(enabled=True))

    with pytest.raises(ControlNetClientError, match="invalid JSON"):
        client._request_json("GET", "/controlnet/version")


def test_controlnet_client_render_outline_rejects_invalid_image_bytes() -> None:
    settings = ControlNetSettings(enabled=True, preprocessor="lineart")
    client = ControlNetClient(settings)

    def fake_request_json(method: str, endpoint: str, payload=None):
        del method, payload
        if endpoint == "/controlnet/version":
            return {"version": 1}
        if endpoint == "/controlnet/module_list":
            return {"module_list": ["lineart"]}
        if endpoint == "/controlnet/model_list":
            return {"model_list": ["control_v11p_sd15_lineart [hash]"]}
        if endpoint == "/sdapi/v1/img2img":
            return {"images": [base64.b64encode(b"not-a-real-image").decode("utf-8")]}
        raise AssertionError(endpoint)

    client._request_json = fake_request_json  # type: ignore[method-assign]

    image = np.full((128, 128, 3), 255, dtype=np.uint8)
    with pytest.raises(ControlNetClientError, match="could not decode"):
        client.render_outline(image)
