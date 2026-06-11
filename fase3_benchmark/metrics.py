import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from typing import Optional, Tuple


def compute_psnr(original: np.ndarray, restored: np.ndarray) -> float:
    if original.shape != restored.shape:
        restored = cv2.resize(restored, (original.shape[1], original.shape[0]))
    return float(psnr(original, restored))


def compute_ssim(original: np.ndarray, restored: np.ndarray) -> float:
    if original.shape != restored.shape:
        restored = cv2.resize(restored, (original.shape[1], original.shape[0]))

    gray_orig = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY) if len(original.shape) == 3 else original
    gray_rest = cv2.cvtColor(restored, cv2.COLOR_BGR2GRAY) if len(restored.shape) == 3 else restored
    return float(ssim(gray_orig, gray_rest))


def compute_sharpness(image: np.ndarray) -> float:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def compute_iou(
    predicted_mask: np.ndarray, ground_truth: np.ndarray
) -> float:
    pred = (predicted_mask > 0).astype(np.uint8)
    gt = (ground_truth > 0).astype(np.uint8)
    intersection = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    return float(intersection / (union + 1e-8))


def compute_precision_recall(
    predicted_mask: np.ndarray, ground_truth: np.ndarray
) -> Tuple[float, float, float]:
    pred = (predicted_mask > 0).astype(bool)
    gt = (ground_truth > 0).astype(bool)

    tp = np.logical_and(pred, gt).sum()
    fp = np.logical_and(pred, ~gt).sum()
    fn = np.logical_and(~pred, gt).sum()

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    return float(precision), float(recall), float(f1)


def compute_restoration_metrics(
    original: np.ndarray,
    degraded: np.ndarray,
    restored: np.ndarray,
) -> dict:
    return {
        "psnr_degraded": compute_psnr(original, degraded),
        "psnr_restored": compute_psnr(original, restored),
        "ssim_degraded": compute_ssim(original, degraded),
        "ssim_restored": compute_ssim(original, restored),
        "sharpness_original": compute_sharpness(original),
        "sharpness_degraded": compute_sharpness(degraded),
        "sharpness_restored": compute_sharpness(restored),
        "psnr_gain": compute_psnr(original, restored) - compute_psnr(original, degraded),
        "ssim_gain": compute_ssim(original, restored) - compute_ssim(original, degraded),
    }


def compute_detection_metrics(
    predicted_mask: np.ndarray,
    ground_truth: np.ndarray,
) -> dict:
    precision, recall, f1 = compute_precision_recall(predicted_mask, ground_truth)
    return {
        "iou": compute_iou(predicted_mask, ground_truth),
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
    }


def feasibility_check(
    fps: float, target_fps: float = 10.0, conveyor_speed_m_min: float = 17.0
) -> dict:
    meets_target = fps >= target_fps
    max_speed = (fps / target_fps) * conveyor_speed_m_min if target_fps > 0 else float("inf")

    return {
        "measured_fps": fps,
        "target_fps": target_fps,
        "meets_requirement": meets_target,
        "max_conveyor_speed_m_min": max_speed,
        "headroom_pct": ((fps - target_fps) / target_fps * 100) if target_fps > 0 else float("inf"),
    }
