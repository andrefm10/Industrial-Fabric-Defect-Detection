import numpy as np
import cv2
from pathlib import Path
from typing import Optional, List, Tuple

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


if TORCH_AVAILABLE:
    class LightCNN(nn.Module):
        def __init__(self, num_classes: int = 5, input_size: int = 64):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(1, 16, 3, padding=1),
                nn.BatchNorm2d(16),
                nn.ReLU(),
                nn.MaxPool2d(2),

                nn.Conv2d(16, 32, 3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.MaxPool2d(2),

                nn.Conv2d(32, 64, 3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((4, 4)),
            )
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(64 * 4 * 4, 128),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(128, num_classes),
            )

        def forward(self, x):
            x = self.features(x)
            return self.classifier(x)


class DefectClassifier:
    LABELS = ["normal", "fold", "dirt", "grammage", "brightness"]

    def __init__(
        self,
        patch_size: int = 64,
        model_path: Optional[Path] = None,
    ):
        self.patch_size = patch_size
        self.model = None

        if TORCH_AVAILABLE:
            self.model = LightCNN(num_classes=len(self.LABELS), input_size=patch_size)
            if model_path and model_path.exists():
                self.model.load_state_dict(torch.load(model_path, weights_only=True))
                self.model.eval()

    def _prepare_patch(self, patch: np.ndarray) -> np.ndarray:
        if len(patch.shape) == 3:
            patch = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(patch, (self.patch_size, self.patch_size))
        return resized.astype(np.float32) / 255.0

    def _augment(self, patch: np.ndarray) -> List[np.ndarray]:
        augmented = [patch]
        augmented.append(np.fliplr(patch))
        augmented.append(np.flipud(patch))
        augmented.append(np.rot90(patch, 1))
        augmented.append(np.rot90(patch, 2))

        noise = patch + np.random.randn(*patch.shape).astype(np.float32) * 0.02
        augmented.append(np.clip(noise, 0, 1))

        bright = np.clip(patch * np.random.uniform(0.8, 1.2), 0, 1).astype(np.float32)
        augmented.append(bright)
        return augmented

    def train(
        self,
        patches_by_label: dict[str, List[np.ndarray]],
        epochs: int = 30,
        lr: float = 1e-3,
        batch_size: int = 32,
    ):
        if not TORCH_AVAILABLE:
            print("PyTorch not available. Classifier disabled.")
            return

        all_patches = []
        all_labels = []

        for label_name, patches in patches_by_label.items():
            if label_name not in self.LABELS:
                continue
            label_idx = self.LABELS.index(label_name)

            for patch in patches:
                prepared = self._prepare_patch(patch)
                augmented = self._augment(prepared)
                for aug in augmented:
                    all_patches.append(aug)
                    all_labels.append(label_idx)

        X = torch.FloatTensor(np.array(all_patches)).unsqueeze(1)
        y = torch.LongTensor(all_labels)

        dataset = TensorDataset(X, y)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        self.model.train()
        for epoch in range(epochs):
            total_loss, correct, total = 0, 0, 0
            for batch_x, batch_y in loader:
                optimizer.zero_grad()
                output = self.model(batch_x)
                loss = criterion(output, batch_y)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                predicted = output.argmax(dim=1)
                correct += (predicted == batch_y).sum().item()
                total += batch_y.size(0)

            if (epoch + 1) % 10 == 0:
                acc = correct / total * 100
                print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss/len(loader):.4f} | Acc: {acc:.1f}%")

        self.model.eval()

    def predict(self, patch: np.ndarray) -> Tuple[str, float]:
        if not TORCH_AVAILABLE or self.model is None:
            return "unknown", 0.0

        prepared = self._prepare_patch(patch)
        tensor = torch.FloatTensor(prepared).unsqueeze(0).unsqueeze(0)

        with torch.no_grad():
            output = self.model(tensor)
            probs = torch.softmax(output, dim=1).squeeze()
            pred_idx = probs.argmax().item()
            confidence = probs[pred_idx].item()

        return self.LABELS[pred_idx], confidence

    def predict_image(
        self, image: np.ndarray, stride: Optional[int] = None
    ) -> Tuple[np.ndarray, List[dict]]:
        h, w = image.shape[:2]
        stride = stride or self.patch_size // 2
        predictions = []

        label_map = np.zeros((h, w), dtype=np.uint8)
        confidence_map = np.zeros((h, w), dtype=np.float32)

        for y in range(0, h - self.patch_size + 1, stride):
            for x in range(0, w - self.patch_size + 1, stride):
                patch = image[y : y + self.patch_size, x : x + self.patch_size]
                label, conf = self.predict(patch)

                if label != "normal":
                    predictions.append({
                        "label": label,
                        "confidence": conf,
                        "bbox": (x, y, self.patch_size, self.patch_size),
                    })

                label_idx = self.LABELS.index(label)
                label_map[y : y + self.patch_size, x : x + self.patch_size] = label_idx
                conf_region = confidence_map[y : y + self.patch_size, x : x + self.patch_size]
                update_mask = conf > conf_region
                confidence_map[y : y + self.patch_size, x : x + self.patch_size] = np.where(
                    update_mask, conf, conf_region
                )

        return label_map, predictions

    def save_model(self, path: Path):
        if TORCH_AVAILABLE and self.model is not None:
            torch.save(self.model.state_dict(), path)
