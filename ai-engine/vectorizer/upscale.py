from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

try:  # pragma: no cover - optional heavyweight dependency
    from basicsr.archs.rrdbnet_arch import RRDBNet
    from realesrgan import RealESRGANer
except ImportError:  # pragma: no cover - optional at runtime
    RRDBNet = None
    RealESRGANer = None


class Upscaler:
    def __init__(self, models_dir: str | Path):
        self.models_dir = Path(models_dir)
        self.model_path = self.models_dir / "RealESRGAN_x4plus.pth"
        self.upsampler = None
        self.model_loaded = False
        self.load_error = None

    def health(self):
        return {
            "modelPath": str(self.model_path),
            "modelExists": self.model_path.exists(),
            "realesrganAvailable": RealESRGANer is not None and RRDBNet is not None,
        }

    def _load_model(self):
        if self.model_loaded or self.load_error:
            return

        if RealESRGANer is None or RRDBNet is None:
            self.load_error = "Real-ESRGAN dependencies are not installed."
            return

        if not self.model_path.exists():
            self.load_error = f"Model weight not found at {self.model_path}."
            return

        try:
            model = RRDBNet(
                num_in_ch=3,
                num_out_ch=3,
                num_feat=64,
                num_block=23,
                num_grow_ch=32,
                scale=4,
            )
            self.upsampler = RealESRGANer(
                scale=4,
                model_path=str(self.model_path),
                model=model,
                tile=0,
                tile_pad=10,
                pre_pad=0,
                half=False,
                gpu_id=None,
            )
            self.model_loaded = True
        except Exception as error:  # pragma: no cover - external library failure
            self.load_error = str(error)

    def _classical_upscale(self, image):
        upscaled = cv2.resize(image, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
        blurred = cv2.GaussianBlur(upscaled, (0, 0), 1.2)
        sharpened = cv2.addWeighted(upscaled, 1.35, blurred, -0.35, 0)
        return np.clip(sharpened, 0, 255).astype(np.uint8)

    def upscale(self, image):
        self._load_model()

        meta = {
            "engine": "classical",
            "fallbackUsed": True,
            "fallbackReason": self.load_error,
        }

        if self.model_loaded and self.upsampler is not None:
            try:  # pragma: no cover - depends on external runtime
                output, _ = self.upsampler.enhance(image, outscale=4)
                meta["engine"] = "realesrgan"
                meta["fallbackUsed"] = False
                meta["fallbackReason"] = None
                return output, meta
            except Exception as error:
                meta["fallbackReason"] = str(error)

        return self._classical_upscale(image), meta
