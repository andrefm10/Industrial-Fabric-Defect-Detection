import cv2
import numpy as np
from typing import Tuple


class CLAHEEnhancer:
    def __init__(
        self,
        clip_limit: float = 3.0,
        grid_size: Tuple[int, int] = (8, 8),
        sharpening_strength: float = 1.0,
        color_space: str = "LAB",
    ):
        self.clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
        self.sharpening_strength = sharpening_strength
        self.color_space = color_space.upper()

    def _get_sharpening_kernel(self) -> np.ndarray:
        base = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]], dtype=np.float64)
        identity = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=np.float64)
        return identity + self.sharpening_strength * (base - identity)

    def _split_luminance(self, image: np.ndarray):
        if self.color_space == "LAB":
            converted = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            channels = list(cv2.split(converted))
            return converted, channels, 0
        elif self.color_space == "YCrCb":
            converted = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
            channels = list(cv2.split(converted))
            return converted, channels, 0
        elif self.color_space == "HSV":
            converted = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            channels = list(cv2.split(converted))
            return converted, channels, 2

    def _merge_back(self, channels: list) -> np.ndarray:
        merged = cv2.merge(channels)
        if self.color_space == "LAB":
            return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
        elif self.color_space == "YCrCb":
            return cv2.cvtColor(merged, cv2.COLOR_YCrCb2BGR)
        elif self.color_space == "HSV":
            return cv2.cvtColor(merged, cv2.COLOR_HSV2BGR)

    def enhance(self, image: np.ndarray) -> np.ndarray:
        _, channels, lum_idx = self._split_luminance(image)
        channels[lum_idx] = self.clahe.apply(channels[lum_idx])

        if self.sharpening_strength > 0:
            kernel = self._get_sharpening_kernel()
            channels[lum_idx] = cv2.filter2D(channels[lum_idx], -1, kernel)

        return self._merge_back(channels)


class BrightnessCorrector:
    def __init__(self, target_mean: float = 127.0, tolerance: float = 30.0):
        self.target_mean = target_mean
        self.tolerance = tolerance

    def needs_correction(self, image: np.ndarray) -> bool:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        return abs(gray.mean() - self.target_mean) > self.tolerance

    def correct(self, image: np.ndarray) -> np.ndarray:
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)

        current_mean = l.mean()
        shift = self.target_mean - current_mean
        l = np.clip(l.astype(np.float64) + shift, 0, 255).astype(np.uint8)

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)

        return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)

    def auto_correct(self, image: np.ndarray) -> np.ndarray:
        if self.needs_correction(image):
            return self.correct(image)
        return image
