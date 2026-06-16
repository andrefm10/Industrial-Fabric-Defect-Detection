from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Tuple

BASE_DIR = Path(__file__).parent
RESULTS_DIR = BASE_DIR / "resultados"
FRAMES_DIR = RESULTS_DIR / "frames"
RESTORED_DIR = RESULTS_DIR / "restaurados"
DEFECTS_DIR = RESULTS_DIR / "defeitos"
BENCHMARKS_DIR = RESULTS_DIR / "benchmarks"
DATASET_DIR = BASE_DIR / "aitex-fabric-image-database"
VIDEOS_DIR = BASE_DIR / "videos"
MODELS_DIR = RESULTS_DIR / "models"

for d in [FRAMES_DIR, RESTORED_DIR, DEFECTS_DIR, BENCHMARKS_DIR, MODELS_DIR]:
    d.mkdir(parents=True, exist_ok=True)


@dataclass
class ConveyorConfig:
    speed_m_per_min: float = 17.0
    camera_fps: float = 30.0
    resolution_px_per_mm: float = 5.0

    @property
    def speed_mm_per_frame(self) -> float:
        speed_mm_per_sec = (self.speed_m_per_min * 1000) / 60
        return speed_mm_per_sec / self.camera_fps

    @property
    def blur_length_px(self) -> int:
        return max(1, int(self.speed_mm_per_frame * self.resolution_px_per_mm))


@dataclass
class RestorationConfig:
    clahe_clip_limit: float = 3.0
    clahe_grid_size: Tuple[int, int] = (8, 8)
    sharpening_strength: float = 1.0

    wiener_snr: float = 0.01
    wiener_blur_angle: float = 0.0

    fft_cutoff_low: float = 30.0
    fft_cutoff_high: float = 100.0

    slice_height_px: int = 4
    slice_overlap: int = 1

    multiframe_window: int = 5


@dataclass
class DetectionConfig:
    gabor_orientations: List[float] = field(
        default_factory=lambda: [0, 45, 90, 135]
    )
    gabor_frequencies: List[float] = field(
        default_factory=lambda: [0.05, 0.1, 0.15, 0.25]
    )
    gabor_sigma: float = 3.0

    otsu_blur_kernel: int = 5
    min_defect_area_px: int = 100

    canny_sigma: float = 1.0
    morphology_kernel_size: int = 5

    brightness_target_mean: float = 127.0
    brightness_tolerance: float = 30.0

    defect_labels: List[str] = field(
        default_factory=lambda: ["normal", "fold", "dirt", "grammage", "brightness"]
    )


@dataclass
class DeepLearningConfig:
    """Parâmetros para treino dos modelos DL (U-Net, CNN)."""
    unet_epochs: int = 20
    unet_lr: float = 1e-3
    unet_batch_size: int = 16
    unet_patch_size: int = 128
    unet_val_split: float = 0.2
    unet_patience: int = 7
    unet_threshold: float = 0.5
    max_normal_images: int = 10


@dataclass
class BenchmarkConfig:
    warmup_iterations: int = 3
    measure_iterations: int = 10
    target_fps: float = 10.0


@dataclass
class AppConfig:
    conveyor: ConveyorConfig = field(default_factory=ConveyorConfig)
    restoration: RestorationConfig = field(default_factory=RestorationConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    deep_learning: DeepLearningConfig = field(default_factory=DeepLearningConfig)
    benchmark: BenchmarkConfig = field(default_factory=BenchmarkConfig)
