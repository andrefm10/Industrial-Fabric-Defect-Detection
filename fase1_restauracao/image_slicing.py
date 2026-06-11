import cv2
import numpy as np


class ImageSlicer:
    def __init__(
        self,
        slice_height: int = 4,
        overlap: int = 1,
        displacement_px: float = 2.0,
        direction: str = "horizontal",
    ):
        self.slice_height = slice_height
        self.overlap = overlap
        self.displacement_px = displacement_px
        self.direction = direction

    def slice_image(self, image: np.ndarray) -> list[np.ndarray]:
        h, w = image.shape[:2]
        slices = []
        step = max(1, self.slice_height - self.overlap)

        for y in range(0, h - self.slice_height + 1, step):
            slices.append(image[y : y + self.slice_height, :].copy())

        return slices

    def reconstruct_aligned(self, image: np.ndarray) -> np.ndarray:
        h, w = image.shape[:2]
        step = max(1, self.slice_height - self.overlap)
        slices = self.slice_image(image)

        canvas_h = len(slices) * step + self.slice_height
        canvas = np.zeros((canvas_h, w, 3) if len(image.shape) == 3 else (canvas_h, w), dtype=np.float64)
        weight = np.zeros((canvas_h, w), dtype=np.float64)

        for i, s in enumerate(slices):
            shift = self.displacement_px * i
            shift_int = int(shift)
            shift_frac = shift - shift_int

            if shift_int > 0 and shift_int < w:
                shifted = np.roll(s, -shift_int, axis=1)
                if shift_frac > 0:
                    shifted_next = np.roll(s, -(shift_int + 1), axis=1)
                    shifted = (1 - shift_frac) * shifted.astype(np.float64) + shift_frac * shifted_next.astype(np.float64)
                else:
                    shifted = shifted.astype(np.float64)
            else:
                shifted = s.astype(np.float64)

            y_start = i * step
            y_end = y_start + self.slice_height
            if y_end <= canvas_h:
                canvas[y_start:y_end] += shifted
                weight[y_start:y_end] += 1.0

        weight = np.maximum(weight, 1.0)
        if len(canvas.shape) == 3:
            canvas /= weight[:, :, None]
        else:
            canvas /= weight

        result = np.clip(canvas[:h, :w], 0, 255).astype(np.uint8)
        return result

    def restore(self, image: np.ndarray) -> np.ndarray:
        return self.reconstruct_aligned(image)
