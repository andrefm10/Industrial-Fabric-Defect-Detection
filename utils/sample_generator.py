import cv2
import numpy as np
from typing import Tuple, Optional


def generate_fabric_texture(
    size: Tuple[int, int] = (512, 512),
    thread_spacing: int = 8,
    thread_width: int = 2,
    base_color: Tuple[int, int, int] = (200, 190, 170),
    noise_level: float = 10.0,
) -> np.ndarray:
    h, w = size
    image = np.full((h, w, 3), base_color, dtype=np.uint8)

    warp_variation = np.random.randint(-15, 15, (h // thread_spacing + 1, 3))
    weft_variation = np.random.randint(-15, 15, (w // thread_spacing + 1, 3))

    for i in range(0, h, thread_spacing):
        idx = i // thread_spacing
        color = np.clip(np.array(base_color) + warp_variation[idx], 0, 255).astype(np.uint8)
        cv2.line(image, (0, i), (w, i), color.tolist(), thread_width)

    for j in range(0, w, thread_spacing):
        idx = j // thread_spacing
        color = np.clip(np.array(base_color) + weft_variation[idx], 0, 255).astype(np.uint8)
        cv2.line(image, (j, 0), (j, h), color.tolist(), thread_width)

    noise = np.random.randn(h, w, 3) * noise_level
    image = np.clip(image.astype(np.float64) + noise, 0, 255).astype(np.uint8)
    return image


def inject_fold_defect(
    image: np.ndarray,
    position: Optional[Tuple[int, int]] = None,
    length: int = 120,
    angle: float = 30.0,
) -> Tuple[np.ndarray, np.ndarray]:
    h, w = image.shape[:2]
    result = image.copy()
    mask = np.zeros((h, w), dtype=np.uint8)

    if position is None:
        position = (np.random.randint(w // 4, 3 * w // 4), np.random.randint(h // 4, 3 * h // 4))

    rad = np.radians(angle)
    dx, dy = int(length * np.cos(rad)), int(length * np.sin(rad))
    pt1 = (position[0] - dx // 2, position[1] - dy // 2)
    pt2 = (position[0] + dx // 2, position[1] + dy // 2)

    cv2.line(mask, pt1, pt2, 255, thickness=6)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.dilate(mask, kernel, iterations=1)

    shadow = np.full_like(result, (50, 45, 40))
    result = np.where(mask[:, :, None] > 0, cv2.addWeighted(result, 0.5, shadow, 0.5, 0), result)
    return result, mask


def inject_dirt_defect(
    image: np.ndarray,
    position: Optional[Tuple[int, int]] = None,
    radius: int = 25,
) -> Tuple[np.ndarray, np.ndarray]:
    h, w = image.shape[:2]
    result = image.copy()
    mask = np.zeros((h, w), dtype=np.uint8)

    if position is None:
        position = (np.random.randint(radius, w - radius), np.random.randint(radius, h - radius))

    cv2.circle(mask, position, radius, 255, -1)
    noise = np.random.randint(0, 60, (h, w, 3), dtype=np.uint8)
    dark_spot = cv2.subtract(result, noise)
    result = np.where(mask[:, :, None] > 0, dark_spot, result)
    result = cv2.GaussianBlur(result, (5, 5), 0)
    return result, mask


def inject_grammage_defect(
    image: np.ndarray,
    region: Optional[Tuple[int, int, int, int]] = None,
    intensity: float = 0.6,
) -> Tuple[np.ndarray, np.ndarray]:
    h, w = image.shape[:2]
    result = image.copy()
    mask = np.zeros((h, w), dtype=np.uint8)

    if region is None:
        rw, rh = np.random.randint(60, 150), np.random.randint(60, 150)
        rx = np.random.randint(0, w - rw)
        ry = np.random.randint(0, h - rh)
        region = (rx, ry, rw, rh)

    rx, ry, rw, rh = region
    mask[ry : ry + rh, rx : rx + rw] = 255

    patch = result[ry : ry + rh, rx : rx + rw].astype(np.float64)
    patch *= intensity
    thin_noise = np.random.randn(rh, rw, 3) * 20
    patch = np.clip(patch + thin_noise, 0, 255).astype(np.uint8)
    result[ry : ry + rh, rx : rx + rw] = patch
    return result, mask


def apply_motion_blur(
    image: np.ndarray, kernel_size: int = 25, angle: float = 0.0
) -> np.ndarray:
    kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
    center = kernel_size // 2
    rad = np.radians(angle)

    for i in range(kernel_size):
        offset = i - center
        x = int(center + offset * np.cos(rad))
        y = int(center + offset * np.sin(rad))
        if 0 <= x < kernel_size and 0 <= y < kernel_size:
            kernel[y, x] = 1.0

    kernel /= kernel.sum()
    return cv2.filter2D(image, -1, kernel)


def generate_test_sample(
    size: Tuple[int, int] = (512, 512),
    blur_length: int = 15,
    defects: bool = True,
) -> dict:
    clean = generate_fabric_texture(size)
    defect_masks = {}

    if defects:
        sample, fold_mask = inject_fold_defect(clean.copy())
        sample, dirt_mask = inject_dirt_defect(sample)
        sample, gram_mask = inject_grammage_defect(sample)
        combined_mask = np.maximum(np.maximum(fold_mask, dirt_mask), gram_mask)
    else:
        sample = clean.copy()
        combined_mask = np.zeros(size, dtype=np.uint8)
        fold_mask = dirt_mask = gram_mask = combined_mask

    blurred = apply_motion_blur(sample, kernel_size=blur_length, angle=0)

    return {
        "clean": clean,
        "defective": sample,
        "blurred": blurred,
        "masks": {
            "fold": fold_mask,
            "dirt": dirt_mask,
            "grammage": gram_mask,
            "combined": combined_mask,
        },
    }
