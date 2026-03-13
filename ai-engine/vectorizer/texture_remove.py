from __future__ import annotations

import cv2
import numpy as np

try:
    from skimage.restoration import denoise_tv_chambolle
except ImportError:  # pragma: no cover - optional at runtime
    denoise_tv_chambolle = None


def _frequency_smooth(channel: np.ndarray) -> np.ndarray:
    height, width = channel.shape
    fft = np.fft.fftshift(np.fft.fft2(channel))
    y, x = np.ogrid[:height, :width]
    center_y = height / 2
    center_x = width / 2
    sigma = max(min(height, width) / 7.5, 8.0)
    gaussian = np.exp(-(((x - center_x) ** 2) + ((y - center_y) ** 2)) / (2 * sigma * sigma))
    filtered = fft * gaussian
    restored = np.fft.ifft2(np.fft.ifftshift(filtered))
    return np.clip(np.real(restored), 0, 255).astype(np.uint8)


def remove_texture(image):
    meta = {"tvDenoise": False}

    median = cv2.medianBlur(image, 3)
    bilateral = cv2.bilateralFilter(median, d=7, sigmaColor=40, sigmaSpace=40)

    lab = cv2.cvtColor(bilateral, cv2.COLOR_BGR2LAB)
    lightness, a_channel, b_channel = cv2.split(lab)
    smoothed_lightness = _frequency_smooth(lightness)

    if denoise_tv_chambolle is not None:
        tv = denoise_tv_chambolle(smoothed_lightness.astype(np.float32) / 255.0, weight=0.07)
        smoothed_lightness = np.clip(tv * 255.0, 0, 255).astype(np.uint8)
        meta["tvDenoise"] = True

    merged = cv2.merge((smoothed_lightness, a_channel, b_channel))
    denoised = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

    if hasattr(cv2, "edgePreservingFilter"):
        edge_preserved = cv2.edgePreservingFilter(denoised, flags=1, sigma_s=45, sigma_r=0.25)
    else:
        edge_preserved = denoised

    blended = cv2.addWeighted(edge_preserved, 0.65, bilateral, 0.35, 0)
    return blended, meta

