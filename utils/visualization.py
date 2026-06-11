import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Optional, Tuple


def show_comparison(
    images: List[np.ndarray],
    titles: List[str],
    save_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (16, 8),
    cmap: Optional[str] = None,
):
    n = len(images)
    fig, axes = plt.subplots(1, n, figsize=figsize)
    if n == 1:
        axes = [axes]

    for ax, img, title in zip(axes, images, titles):
        display = img
        if len(img.shape) == 3 and img.shape[2] == 3:
            display = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ax.imshow(display, cmap=cmap if len(img.shape) == 2 else None)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.axis("off")

    plt.tight_layout()
    if save_path:
        fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.close(fig)


def show_grid(
    images: List[np.ndarray],
    titles: List[str],
    cols: int = 3,
    save_path: Optional[Path] = None,
    figsize_per_image: Tuple[int, int] = (5, 5),
):
    n = len(images)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(
        rows, cols, figsize=(figsize_per_image[0] * cols, figsize_per_image[1] * rows)
    )
    axes = np.array(axes).flatten()

    for i, (ax, img, title) in enumerate(zip(axes, images, titles)):
        display = img
        if len(img.shape) == 3 and img.shape[2] == 3:
            display = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ax.imshow(display, cmap="gray" if len(img.shape) == 2 else None)
        ax.set_title(title, fontsize=10)
        ax.axis("off")

    for j in range(n, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    if save_path:
        fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
    plt.close(fig)


def overlay_heatmap(
    image: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.5,
    colormap: int = cv2.COLORMAP_JET,
) -> np.ndarray:
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    heatmap_normalized = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX).astype(
        np.uint8
    )
    colored_heatmap = cv2.applyColorMap(heatmap_normalized, colormap)
    return cv2.addWeighted(image, 1 - alpha, colored_heatmap, alpha, 0)


def draw_defect_contours(
    image: np.ndarray,
    mask: np.ndarray,
    color: Tuple[int, int, int] = (0, 0, 255),
    thickness: int = 2,
    label: str = "",
) -> np.ndarray:
    output = image.copy()
    if len(output.shape) == 2:
        output = cv2.cvtColor(output, cv2.COLOR_GRAY2BGR)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(output, contours, -1, color, thickness)

    if label:
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.putText(
                output, label, (x, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1,
            )
    return output


def plot_histogram_comparison(
    original: np.ndarray,
    processed: np.ndarray,
    save_path: Optional[Path] = None,
):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    for ax, img, title in [(ax1, original, "Original"), (ax2, processed, "Processed")]:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
        ax.hist(gray.ravel(), bins=256, range=(0, 256), color="steelblue", alpha=0.7)
        ax.set_title(f"Histogram — {title}")
        ax.set_xlabel("Intensity")
        ax.set_ylabel("Count")
        ax.axvline(gray.mean(), color="red", linestyle="--", label=f"Mean: {gray.mean():.1f}")
        ax.legend()

    plt.tight_layout()
    if save_path:
        fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_fft_spectrum(
    image: np.ndarray, save_path: Optional[Path] = None
) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    f_transform = np.fft.fft2(gray.astype(np.float64))
    f_shift = np.fft.fftshift(f_transform)
    magnitude = np.log1p(np.abs(f_shift))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    ax1.imshow(gray, cmap="gray")
    ax1.set_title("Spatial Domain")
    ax1.axis("off")
    ax2.imshow(magnitude, cmap="inferno")
    ax2.set_title("Frequency Spectrum (Log Magnitude)")
    ax2.axis("off")

    plt.tight_layout()
    if save_path:
        fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    return magnitude
