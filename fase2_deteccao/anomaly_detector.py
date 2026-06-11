import numpy as np
import cv2
from pathlib import Path
from typing import Optional, Tuple

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


if TORCH_AVAILABLE:
    class ConvAutoencoder(nn.Module):
        def __init__(self, input_channels: int = 1):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Conv2d(input_channels, 32, 3, stride=2, padding=1),
                nn.ReLU(),
                nn.Conv2d(32, 64, 3, stride=2, padding=1),
                nn.ReLU(),
                nn.Conv2d(64, 128, 3, stride=2, padding=1),
                nn.ReLU(),
            )
            self.decoder = nn.Sequential(
                nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
                nn.ReLU(),
                nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
                nn.ReLU(),
                nn.ConvTranspose2d(32, input_channels, 3, stride=2, padding=1, output_padding=1),
                nn.Sigmoid(),
            )

        def forward(self, x):
            encoded = self.encoder(x)
            decoded = self.decoder(encoded)
            return decoded


class AnomalyDetector:
    def __init__(
        self,
        patch_size: int = 64,
        threshold_percentile: float = 95.0,
        model_path: Optional[Path] = None,
    ):
        self.patch_size = patch_size
        self.threshold_percentile = threshold_percentile
        self.model = None
        self.threshold = None
        self._model_trained = False

        if TORCH_AVAILABLE:
            self.model = ConvAutoencoder(input_channels=1)
            if model_path and model_path.exists():
                self.model.load_state_dict(torch.load(model_path, weights_only=True))
                self.model.eval()
                self._model_trained = True

    def _extract_patches(
        self, image: np.ndarray, stride: Optional[int] = None
    ) -> np.ndarray:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        stride = stride or self.patch_size // 2
        h, w = gray.shape
        patches = []

        for y in range(0, h - self.patch_size + 1, stride):
            for x in range(0, w - self.patch_size + 1, stride):
                patch = gray[y : y + self.patch_size, x : x + self.patch_size]
                patches.append(patch)

        return np.array(patches, dtype=np.float32) / 255.0

    def train(
        self,
        normal_images: list[np.ndarray],
        epochs: int = 20,
        lr: float = 1e-3,
        batch_size: int = 32,
        max_patches_per_image: int = 200,
    ):
        if not TORCH_AVAILABLE:
            print("PyTorch not available. Using statistical fallback.")
            self._train_statistical(normal_images)
            return

        all_patches = []
        for img in normal_images:
            patches = self._extract_patches(img, stride=self.patch_size)
            # Subsample if too many patches (large images)
            if len(patches) > max_patches_per_image:
                indices = np.random.choice(len(patches), max_patches_per_image, replace=False)
                patches = patches[indices]
            all_patches.append(patches)

        patches = np.concatenate(all_patches, axis=0)
        print(f"  Total training patches: {len(patches)}")
        tensor = torch.FloatTensor(patches).unsqueeze(1)
        dataset = TensorDataset(tensor, tensor)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.MSELoss()

        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            for batch_x, batch_y in loader:
                optimizer.zero_grad()
                output = self.model(batch_x)
                h_out, w_out = output.shape[2], output.shape[3]
                batch_y_cropped = batch_y[:, :, :h_out, :w_out]
                loss = criterion(output, batch_y_cropped)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss/len(loader):.6f}")

        self.model.eval()
        self._model_trained = True
        self._calibrate_threshold(normal_images)

    def _train_statistical(self, normal_images: list[np.ndarray]):
        all_patches = []
        for img in normal_images:
            patches = self._extract_patches(img)
            all_patches.append(patches)
        patches = np.concatenate(all_patches, axis=0)
        self._stat_mean = patches.mean(axis=0)
        self._stat_std = patches.std(axis=0) + 1e-8
        errors = np.mean((patches - self._stat_mean) ** 2, axis=(1, 2))
        self.threshold = np.percentile(errors, self.threshold_percentile)

    def _calibrate_threshold(self, normal_images: list[np.ndarray]):
        errors = []
        for img in normal_images:
            error_map = self._compute_error_map(img)
            errors.append(error_map.mean())
        self.threshold = np.percentile(errors, self.threshold_percentile)

    def _compute_error_map(self, image: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        h, w = gray.shape
        error_map = np.zeros((h, w), dtype=np.float64)
        count_map = np.zeros((h, w), dtype=np.float64)
        # Use full patch_size stride for speed on large images
        stride = self.patch_size

        for y in range(0, h - self.patch_size + 1, stride):
            for x in range(0, w - self.patch_size + 1, stride):
                patch = gray[y : y + self.patch_size, x : x + self.patch_size]
                patch_norm = patch.astype(np.float32) / 255.0

                if TORCH_AVAILABLE and self.model is not None and self._model_trained:
                    with torch.no_grad():
                        inp = torch.FloatTensor(patch_norm).unsqueeze(0).unsqueeze(0)
                        out = self.model(inp).squeeze().numpy()
                        out_h, out_w = out.shape
                        error = (patch_norm[:out_h, :out_w] - out) ** 2
                        error_full = np.zeros_like(patch_norm)
                        error_full[:out_h, :out_w] = error
                elif hasattr(self, '_stat_mean'):
                    error_full = (patch_norm - self._stat_mean) ** 2
                else:
                    local_mean = patch_norm.mean()
                    error_full = (patch_norm - local_mean) ** 2

                error_map[y : y + self.patch_size, x : x + self.patch_size] += error_full
                count_map[y : y + self.patch_size, x : x + self.patch_size] += 1

        count_map = np.maximum(count_map, 1)
        return error_map / count_map

    def detect(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        error_map = self._compute_error_map(image)

        if self.threshold is None:
            self.threshold = float(np.percentile(error_map, self.threshold_percentile))

        binary = (error_map > self.threshold).astype(np.uint8) * 255

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

        heatmap = cv2.normalize(error_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        return binary, heatmap

    def save_model(self, path: Path):
        if TORCH_AVAILABLE and self.model is not None:
            torch.save(self.model.state_dict(), path)
            print(f"Model saved: {path}")
