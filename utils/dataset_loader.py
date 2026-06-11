"""
AITEX Fabric Image Database Loader

Loads and organizes the AITEX dataset:
  - Defect_images/   → 106 images with defects (256×4096)
  - Mask_images/     → 107 ground truth masks
  - NODefect_images/ → 141 defect-free images (7 fabric subfolders)

Naming convention: {ID}_{FABRIC}_{DEFECTTYPE}.png
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Generator
from dataclasses import dataclass, field


# AITEX defect type codes → human-readable names
DEFECT_TYPE_MAP = {
    "00": "Broken end",
    "01": "Broken yarn",
    "02": "Cut",
    "03": "Fuzzyball",
    "04": "Pilling",
    "05": "Nep",
    "06": "Weft crack",
    "08": "Contamination",
}


@dataclass
class FabricSample:
    """Represents a single fabric image from the dataset."""
    image_path: Path
    mask_path: Optional[Path]
    fabric_code: str
    defect_code: Optional[str]
    sample_id: str
    has_defect: bool

    @property
    def defect_name(self) -> str:
        if self.defect_code is None:
            return "Normal"
        return DEFECT_TYPE_MAP.get(self.defect_code, f"Unknown ({self.defect_code})")

    def load_image(self) -> np.ndarray:
        img = cv2.imread(str(self.image_path))
        if img is None:
            raise FileNotFoundError(f"Cannot load: {self.image_path}")
        return img

    def load_mask(self) -> Optional[np.ndarray]:
        if self.mask_path is None or not self.mask_path.exists():
            return None
        mask = cv2.imread(str(self.mask_path), cv2.IMREAD_GRAYSCALE)
        return mask


class AitexDataset:
    """Loader for the AITEX Fabric Image Database."""

    def __init__(self, dataset_dir: Path):
        self.root = Path(dataset_dir)
        self.defect_dir = self.root / "Defect_images"
        self.mask_dir = self.root / "Mask_images"
        self.nodefect_dir = self.root / "NODefect_images"

        if not self.root.exists():
            raise FileNotFoundError(f"Dataset not found: {self.root}")

        self._defect_samples: List[FabricSample] = []
        self._normal_samples: List[FabricSample] = []
        self._load_index()

    def _load_index(self):
        """Index all images in the dataset."""
        # Load defect images
        if self.defect_dir.exists():
            for img_path in sorted(self.defect_dir.glob("*.png")):
                parts = img_path.stem.split("_")
                if len(parts) >= 3:
                    sample_id = parts[0]
                    fabric_code = parts[1]
                    defect_code = parts[2]

                    mask_name = f"{img_path.stem}_mask.png"
                    mask_path = self.mask_dir / mask_name

                    self._defect_samples.append(FabricSample(
                        image_path=img_path,
                        mask_path=mask_path if mask_path.exists() else None,
                        fabric_code=fabric_code,
                        defect_code=defect_code,
                        sample_id=sample_id,
                        has_defect=True,
                    ))

        # Load normal (no-defect) images
        if self.nodefect_dir.exists():
            for subdir in sorted(self.nodefect_dir.iterdir()):
                if subdir.is_dir():
                    for img_path in sorted(subdir.glob("*.png")):
                        parts = img_path.stem.split("_")
                        sample_id = parts[0] if parts else img_path.stem
                        fabric_code = parts[1] if len(parts) >= 2 else subdir.name

                        self._normal_samples.append(FabricSample(
                            image_path=img_path,
                            mask_path=None,
                            fabric_code=fabric_code,
                            defect_code=None,
                            sample_id=sample_id,
                            has_defect=False,
                        ))

    @property
    def defect_samples(self) -> List[FabricSample]:
        return self._defect_samples

    @property
    def normal_samples(self) -> List[FabricSample]:
        return self._normal_samples

    @property
    def all_samples(self) -> List[FabricSample]:
        return self._defect_samples + self._normal_samples

    @property
    def num_defect(self) -> int:
        return len(self._defect_samples)

    @property
    def num_normal(self) -> int:
        return len(self._normal_samples)

    @property
    def defect_types(self) -> Dict[str, int]:
        """Count of images per defect type."""
        counts: Dict[str, int] = {}
        for s in self._defect_samples:
            name = s.defect_name
            counts[name] = counts.get(name, 0) + 1
        return dict(sorted(counts.items(), key=lambda x: -x[1]))

    @property
    def fabric_types(self) -> Dict[str, int]:
        """Count of images per fabric type."""
        counts: Dict[str, int] = {}
        for s in self.all_samples:
            counts[s.fabric_code] = counts.get(s.fabric_code, 0) + 1
        return dict(sorted(counts.items(), key=lambda x: -x[1]))

    def get_by_defect_type(self, defect_code: str) -> List[FabricSample]:
        return [s for s in self._defect_samples if s.defect_code == defect_code]

    def get_by_fabric(self, fabric_code: str) -> List[FabricSample]:
        return [s for s in self.all_samples if s.fabric_code == fabric_code]

    def iter_defect_with_masks(self) -> Generator[Tuple[FabricSample, np.ndarray, np.ndarray], None, None]:
        """Yield (sample, image, mask) for all defect images that have masks."""
        for sample in self._defect_samples:
            if sample.mask_path and sample.mask_path.exists():
                image = sample.load_image()
                mask = sample.load_mask()
                if image is not None and mask is not None:
                    yield sample, image, mask

    def load_normal_images(self, max_count: Optional[int] = None) -> List[np.ndarray]:
        """Load normal (defect-free) images for training."""
        images = []
        for sample in self._normal_samples[:max_count]:
            try:
                images.append(sample.load_image())
            except FileNotFoundError:
                continue
        return images

    def summary(self) -> str:
        lines = [
            f"AITEX Dataset: {self.root}",
            f"  Defect images:  {self.num_defect}",
            f"  Normal images:  {self.num_normal}",
            f"  Total:          {self.num_defect + self.num_normal}",
            f"  Defect types:   {len(self.defect_types)}",
            f"  Fabric types:   {len(self.fabric_types)}",
            "",
            "  Defect distribution:",
        ]
        for name, count in self.defect_types.items():
            lines.append(f"    {name}: {count}")
        return "\n".join(lines)
