import cv2
import numpy as np
from typing import Tuple


class HistogramOtsuDetector:
    def __init__(self, blur_kernel: int = 5, min_area: int = 100):
        self.blur_kernel = blur_kernel
        self.min_area = min_area

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        blurred = cv2.GaussianBlur(gray, (self.blur_kernel, self.blur_kernel), 0)
        return blurred

    def otsu_threshold(self, image: np.ndarray) -> Tuple[np.ndarray, float]:
        gray = self.preprocess(image)
        threshold_value, binary = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )
        return binary, threshold_value

    def multi_otsu(self, image: np.ndarray, levels: int = 3) -> np.ndarray:
        from skimage.filters import threshold_multiotsu

        gray = self.preprocess(image)
        thresholds = threshold_multiotsu(gray, classes=levels)
        regions = np.digitize(gray, bins=thresholds)
        return (regions * (255 // (levels - 1))).astype(np.uint8)

    def adaptive_threshold(self, image: np.ndarray) -> np.ndarray:
        gray = self.preprocess(image)
        return cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 21, 5,
        )

    def detect_defects(
        self, image: np.ndarray, method: str = "otsu"
    ) -> Tuple[np.ndarray, list]:
        if method == "otsu":
            binary, _ = self.otsu_threshold(image)
        elif method == "multi_otsu":
            segmented = self.multi_otsu(image)
            binary = cv2.threshold(segmented, segmented.mean(), 255, cv2.THRESH_BINARY)[1]
        elif method == "adaptive":
            binary = self.adaptive_threshold(image)
        else:
            raise ValueError(f"Unknown method: {method}")

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        defects = [c for c in contours if cv2.contourArea(c) >= self.min_area]

        mask = np.zeros_like(cleaned)
        cv2.drawContours(mask, defects, -1, 255, -1)
        return mask, defects

    def analyze_histogram(self, image: np.ndarray) -> dict:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()

        return {
            "mean": float(gray.mean()),
            "std": float(gray.std()),
            "median": float(np.median(gray)),
            "skewness": float(((gray.astype(float) - gray.mean()) ** 3).mean() / (gray.std() ** 3 + 1e-8)),
            "histogram": hist,
            "is_dark": gray.mean() < 80,
            "is_bright": gray.mean() > 200,
            "is_low_contrast": gray.std() < 30,
        }
