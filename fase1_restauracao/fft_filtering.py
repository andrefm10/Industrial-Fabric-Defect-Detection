import cv2
import numpy as np
from typing import Tuple


def compute_fft(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    f = np.fft.fft2(gray.astype(np.float64))
    f_shift = np.fft.fftshift(f)
    magnitude = np.log1p(np.abs(f_shift))
    return f_shift, magnitude


def butterworth_lowpass(shape: Tuple[int, int], cutoff: float, order: int = 2) -> np.ndarray:
    rows, cols = shape
    crow, ccol = rows // 2, cols // 2
    u = np.arange(rows).reshape(-1, 1) - crow
    v = np.arange(cols).reshape(1, -1) - ccol
    d = np.sqrt(u**2 + v**2)
    return 1.0 / (1.0 + (d / cutoff) ** (2 * order))


def butterworth_highpass(shape: Tuple[int, int], cutoff: float, order: int = 2) -> np.ndarray:
    return 1.0 - butterworth_lowpass(shape, cutoff, order)


def bandpass_filter(
    shape: Tuple[int, int], low_cutoff: float, high_cutoff: float, order: int = 2
) -> np.ndarray:
    lp = butterworth_lowpass(shape, high_cutoff, order)
    hp = butterworth_highpass(shape, low_cutoff, order)
    return lp * hp


def notch_filter(
    shape: Tuple[int, int],
    centers: list[Tuple[int, int]],
    radius: float = 10.0,
) -> np.ndarray:
    rows, cols = shape
    mask = np.ones((rows, cols), dtype=np.float64)
    crow, ccol = rows // 2, cols // 2

    for cx, cy in centers:
        u = np.arange(rows).reshape(-1, 1)
        v = np.arange(cols).reshape(1, -1)
        d1 = np.sqrt((u - (crow + cy)) ** 2 + (v - (ccol + cx)) ** 2)
        d2 = np.sqrt((u - (crow - cy)) ** 2 + (v - (ccol - cx)) ** 2)
        mask *= (1.0 - np.exp(-0.5 * (d1 * d2 / (radius**2 + 1e-8))))

    return mask


def apply_frequency_filter(
    image: np.ndarray, filter_mask: np.ndarray
) -> np.ndarray:
    if len(image.shape) == 3:
        channels = cv2.split(image)
        filtered = [_filter_channel(ch, filter_mask) for ch in channels]
        return cv2.merge(filtered)
    return _filter_channel(image, filter_mask)


def _filter_channel(channel: np.ndarray, filter_mask: np.ndarray) -> np.ndarray:
    f = np.fft.fft2(channel.astype(np.float64))
    f_shift = np.fft.fftshift(f)
    filtered = f_shift * filter_mask
    result = np.fft.ifft2(np.fft.ifftshift(filtered))
    return np.clip(np.abs(result), 0, 255).astype(np.uint8)


class FFTRestorer:
    def __init__(
        self,
        low_cutoff: float = 5.0,
        high_cutoff: float = 100.0,
        filter_type: str = "bandpass",
    ):
        self.low_cutoff = low_cutoff
        self.high_cutoff = high_cutoff
        self.filter_type = filter_type

    def restore(self, image: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        shape = gray.shape

        if self.filter_type == "lowpass":
            mask = butterworth_lowpass(shape, self.high_cutoff)
        elif self.filter_type == "highpass":
            mask = butterworth_highpass(shape, self.low_cutoff)
        elif self.filter_type == "bandpass":
            mask = bandpass_filter(shape, self.low_cutoff, self.high_cutoff)
        else:
            raise ValueError(f"Unknown filter type: {self.filter_type}")

        return apply_frequency_filter(image, mask)

    def auto_denoise(self, image: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        f_shift, magnitude = compute_fft(gray)

        rows, cols = gray.shape
        crow, ccol = rows // 2, cols // 2
        center_region = magnitude[
            crow - 5 : crow + 5, ccol - 5 : ccol + 5
        ]
        threshold = center_region.mean() * 0.3

        peaks = []
        for i in range(rows):
            for j in range(cols):
                dist = np.sqrt((i - crow) ** 2 + (j - ccol) ** 2)
                if dist > 20 and magnitude[i, j] > threshold:
                    peaks.append((j - ccol, i - crow))

        if not peaks:
            mask = butterworth_lowpass(gray.shape, self.high_cutoff)
        else:
            mask = notch_filter(gray.shape, peaks[:10], radius=15)

        return apply_frequency_filter(image, mask)
