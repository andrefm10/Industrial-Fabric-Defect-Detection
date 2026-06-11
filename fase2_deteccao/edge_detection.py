import cv2
import numpy as np
from typing import List, Tuple


class EdgeDefectDetector:
    def __init__(
        self,
        canny_sigma: float = 1.0,
        morphology_kernel: int = 5,
        min_area: int = 100,
        max_area_ratio: float = 0.4,
        border_margin: int = 5,
    ):
        self.canny_sigma = canny_sigma
        self.morphology_kernel = morphology_kernel
        self.min_area = min_area
        self.max_area_ratio = max_area_ratio
        self.border_margin = border_margin

    def auto_canny(self, image: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        blurred = cv2.GaussianBlur(gray, (5, 5), self.canny_sigma)

        median = np.median(blurred)
        lower = int(max(0, (1.0 - 0.33) * median))
        upper = int(min(255, (1.0 + 0.33) * median))
        return cv2.Canny(blurred, lower, upper)

    def detect_edges(self, image: np.ndarray) -> np.ndarray:
        edges = self.auto_canny(image)
        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (self.morphology_kernel, self.morphology_kernel)
        )
        closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        return closed

    def classify_contour(self, contour: np.ndarray) -> str:
        area = cv2.contourArea(contour)
        if area < self.min_area:
            return "noise"

        perimeter = cv2.arcLength(contour, True)
        circularity = 4 * np.pi * area / (perimeter ** 2 + 1e-8)

        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = max(w, h) / (min(w, h) + 1e-8)

        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        solidity = area / (hull_area + 1e-8)

        if aspect_ratio > 4.0:
            return "fold"
        elif circularity > 0.6:
            return "dirt"
        elif solidity < 0.5:
            return "grammage"
        return "unknown"

    def _is_frame_border(self, contour: np.ndarray, img_h: int, img_w: int) -> bool:
        x, y, w, h = cv2.boundingRect(contour)
        m = self.border_margin
        return x <= m and y <= m and (x + w) >= (img_w - m) and (y + h) >= (img_h - m)

    def detect_and_classify(
        self, image: np.ndarray
    ) -> Tuple[np.ndarray, List[dict]]:
        img_h, img_w = image.shape[:2]
        frame_area = img_h * img_w
        max_area = frame_area * self.max_area_ratio

        edges = self.detect_edges(image)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        defects = []
        result_mask = np.zeros(image.shape[:2], dtype=np.uint8)

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self.min_area or area > max_area:
                continue
            if self._is_frame_border(contour, img_h, img_w):
                continue

            label = self.classify_contour(contour)
            x, y, w, h = cv2.boundingRect(contour)

            defects.append({
                "label": label,
                "area": area,
                "bbox": (x, y, w, h),
                "contour": contour,
            })
            cv2.drawContours(result_mask, [contour], -1, 255, -1)

        return result_mask, defects

    def annotate_image(
        self, image: np.ndarray, defects: List[dict]
    ) -> np.ndarray:
        output = image.copy()
        if len(output.shape) == 2:
            output = cv2.cvtColor(output, cv2.COLOR_GRAY2BGR)

        colors = {
            "fold": (0, 255, 255),
            "dirt": (0, 0, 255),
            "grammage": (255, 0, 255),
            "unknown": (128, 128, 128),
        }

        for defect in defects:
            color = colors.get(defect["label"], (255, 255, 255))
            x, y, w, h = defect["bbox"]
            cv2.rectangle(output, (x, y), (x + w, y + h), color, 2)
            cv2.putText(
                output,
                f"{defect['label']} ({defect['area']:.0f}px)",
                (x, y - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1,
            )

        return output
