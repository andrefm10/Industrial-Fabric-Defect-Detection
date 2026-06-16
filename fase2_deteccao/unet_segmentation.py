"""
U-Net — Segmentação Semântica de Defeitos Têxteis

Arquitetura U-Net com 4 níveis encoder/decoder:
  - Encoder: Conv3×3 → BatchNorm → ReLU → Conv3×3 → BatchNorm → ReLU → MaxPool2×2
  - Bottleneck: 256 → 512 canais
  - Decoder: ConvTranspose2×2 → Concat(skip) → Conv3×3 → BN → ReLU × 2
  - Saída: Conv1×1 → Sigmoid → mapa de probabilidade [0, 1]

Skip connections preservam detalhes espaciais (bordas dos defeitos)
combinando features do encoder com o decoder em cada nível.

Treino supervisionado com ground truth masks do AITEX.
Loss: BCE + Dice (robusto ao desbalanceamento de classes).
"""

import numpy as np
import cv2
from pathlib import Path
from typing import Optional, Tuple, List

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


# ─── Blocos da Arquitetura ────────────────────────────────────────────────────

if TORCH_AVAILABLE:

    class ConvBlock(nn.Module):
        """
        Bloco básico da U-Net: duas convoluções 3×3 consecutivas.

        Cada convolução é seguida de BatchNorm (estabiliza gradientes)
        e ReLU (introduz não-linearidade — sem ela a rede seria linear).

        Conv3×3 → BatchNorm → ReLU → Conv3×3 → BatchNorm → ReLU
        """
        def __init__(self, in_channels: int, out_channels: int):
            super().__init__()
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )

        def forward(self, x):
            return self.conv(x)

    class UNet(nn.Module):
        """
        U-Net para segmentação binária de defeitos.

        4 níveis de encoder/decoder:
          Encoder:     1 → 32 → 64 → 128 → 256  (MaxPool2×2 entre cada)
          Bottleneck:  256 → 512
          Decoder:     512 → 256 → 128 → 64 → 32  (ConvTranspose2×2)
          Saída:       32 → 1 canal (Sigmoid)

        Skip connections: concatenam features do encoder com o decoder
        no mesmo nível de resolução. Isso permite que a rede combine:
          - Features abstratas (do bottleneck): "O QUE é"
          - Features espaciais (do encoder): "ONDE está"
        """
        def __init__(self, in_channels: int = 1, out_channels: int = 1,
                     features: Optional[List[int]] = None):
            super().__init__()
            if features is None:
                features = [32, 64, 128, 256]

            # Encoder: cada bloco extrai features e reduz resolução
            self.encoder_blocks = nn.ModuleList()
            self.pool = nn.MaxPool2d(2, 2)
            ch = in_channels
            for f in features:
                self.encoder_blocks.append(ConvBlock(ch, f))
                ch = f

            # Bottleneck: representação mais comprimida
            self.bottleneck = ConvBlock(features[-1], features[-1] * 2)

            # Decoder: reconstrói a resolução original
            self.upconvs = nn.ModuleList()
            self.decoder_blocks = nn.ModuleList()
            for f in reversed(features):
                # ConvTranspose2d: "desfaz" o MaxPool — dobra resolução
                self.upconvs.append(
                    nn.ConvTranspose2d(f * 2, f, kernel_size=2, stride=2)
                )
                # f*2 no input porque concatenamos com skip connection
                self.decoder_blocks.append(ConvBlock(f * 2, f))

            # Convolução final 1×1: reduz de 32 canais para 1
            self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

        def forward(self, x):
            # ── Encoder ──
            skip_features = []
            for encoder in self.encoder_blocks:
                x = encoder(x)
                skip_features.append(x)   # Guarda para skip connection
                x = self.pool(x)          # Reduz resolução 2×

            # ── Bottleneck ──
            x = self.bottleneck(x)

            # ── Decoder ──
            skip_features = skip_features[::-1]  # Inverte a ordem
            for i, (upconv, decoder) in enumerate(zip(self.upconvs, self.decoder_blocks)):
                x = upconv(x)             # Dobra resolução 2×
                skip = skip_features[i]

                # Ajusta tamanho se necessário (dimensões não divisíveis por 2)
                if x.shape[2:] != skip.shape[2:]:
                    x = F.interpolate(x, size=skip.shape[2:],
                                      mode="bilinear", align_corners=True)

                # SKIP CONNECTION: concatena features do encoder
                x = torch.cat([skip, x], dim=1)
                x = decoder(x)

            # Sigmoid mapeia output para [0, 1] = probabilidade de defeito
            return torch.sigmoid(self.final_conv(x))

    class DiceBCELoss(nn.Module):
        """
        Loss combinada: BCE + Dice.

        BCE (Binary Cross-Entropy):
          Penaliza cada pixel individualmente.
          -[y·log(p) + (1-y)·log(1-p)]

        Dice Loss:
          Mede sobreposição global entre predição e ground truth.
          1 - (2·|A∩B| + ε) / (|A| + |B| + ε)

          Crucial quando classes são desbalanceadas (defeitos são <5%
          da imagem). BCE sozinho tenderia a prever "tudo normal".
        """
        def __init__(self, dice_weight: float = 0.5, smooth: float = 1.0):
            super().__init__()
            self.dice_weight = dice_weight
            self.bce_weight = 1.0 - dice_weight
            self.smooth = smooth
            self.bce = nn.BCELoss()

        def forward(self, predictions, targets):
            bce_loss = self.bce(predictions, targets)

            pred_flat = predictions.view(-1)
            tgt_flat = targets.view(-1)
            intersection = (pred_flat * tgt_flat).sum()
            dice = (2.0 * intersection + self.smooth) / (
                pred_flat.sum() + tgt_flat.sum() + self.smooth
            )
            dice_loss = 1.0 - dice

            return self.bce_weight * bce_loss + self.dice_weight * dice_loss


# ─── Dataset de Patches para Treino ──────────────────────────────────────────

if TORCH_AVAILABLE:

    class FabricPatchDataset(Dataset):
        """
        Dataset PyTorch que fornece patches 256×256 com augmentation.

        Data augmentation simula variações que a rede veria em produção:
          - Flip horizontal/vertical: tecido pode estar em qualquer orientação
          - Rotação 90°: mesma razão
          - Variação de brilho: simula mudanças de iluminação
          - Ruído Gaussiano: simula ruído do sensor da câmera
        """
        def __init__(self, patches: List[np.ndarray],
                     masks: List[np.ndarray], augment: bool = True):
            self.patches = patches
            self.masks = masks
            self.augment = augment

        def __len__(self):
            return len(self.patches)

        def _augment(self, image: np.ndarray, mask: np.ndarray):
            if np.random.rand() > 0.5:
                image = np.fliplr(image).copy()
                mask = np.fliplr(mask).copy()

            if np.random.rand() > 0.5:
                image = np.flipud(image).copy()
                mask = np.flipud(mask).copy()

            k = np.random.randint(0, 4)
            image = np.rot90(image, k).copy()
            mask = np.rot90(mask, k).copy()

            if np.random.rand() > 0.5:
                factor = np.random.uniform(0.8, 1.2)
                image = np.clip(image * factor, 0, 1)

            if np.random.rand() > 0.5:
                noise = np.random.randn(*image.shape).astype(np.float32) * 0.02
                image = np.clip(image + noise, 0, 1)

            return image.astype(np.float32), mask.astype(np.float32)

        def __getitem__(self, idx):
            image = self.patches[idx].astype(np.float32)
            mask = self.masks[idx].astype(np.float32)

            if self.augment:
                image, mask = self._augment(image, mask)

            return (torch.FloatTensor(image).unsqueeze(0),
                    torch.FloatTensor(mask).unsqueeze(0))


# ─── API de Alto Nível ────────────────────────────────────────────────────────

class UNetSegmenter:
    """
    Wrapper de alto nível para treino e inferência da U-Net.

    Treino:
      1. Extrai patches 256×256 das imagens AITEX (256×4096)
      2. Split treino/validação
      3. Treina com BCE+Dice, early stopping, LR scheduler
      4. Salva melhor modelo (por val loss)

    Inferência:
      1. Desliza janela 256×256 pela imagem (sliding window)
      2. Prediz cada patch
      3. Faz blending das sobreposições (média)
      4. Aplica threshold → máscara binária
    """

    def __init__(self, patch_size: int = 256,
                 model_path: Optional[Path] = None,
                 device: Optional[str] = None):
        self.patch_size = patch_size
        self._trained = False

        if not TORCH_AVAILABLE:
            self.model = None
            self.device = None
            return

        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model = UNet(in_channels=1, out_channels=1,
                          features=[16, 32, 64, 128]).to(self.device)

        if model_path and model_path.exists():
            self.model.load_state_dict(
                torch.load(model_path, map_location=self.device, weights_only=True)
            )
            self.model.eval()
            self._trained = True
            print(f"  U-Net loaded from: {model_path}")

    # ── Extração de Patches ──

    def _extract_patches(
        self, image: np.ndarray, mask: np.ndarray,
        stride: Optional[int] = None,
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Extrai pares (patch, máscara) de uma imagem AITEX."""
        gray = (cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                if len(image.shape) == 3 else image)
        gray = gray.astype(np.float32) / 255.0
        mask_norm = (mask > 0).astype(np.float32)

        h, w = gray.shape
        ps = self.patch_size
        stride = stride or ps // 2

        patches, mask_patches = [], []
        for y in range(0, max(1, h - ps + 1), stride):
            for x in range(0, max(1, w - ps + 1), stride):
                ye, xe = min(y + ps, h), min(x + ps, w)
                ys, xs = ye - ps, xe - ps
                patch = gray[ys:ye, xs:xe]
                m_patch = mask_norm[ys:ye, xs:xe]
                if patch.shape == (ps, ps):
                    patches.append(patch)
                    mask_patches.append(m_patch)

        return patches, mask_patches

    # ── Treino ──

    def train(
        self,
        defect_images: List[np.ndarray],
        defect_masks: List[np.ndarray],
        normal_images: List[np.ndarray],
        epochs: int = 30,
        lr: float = 1e-3,
        batch_size: int = 8,
        val_split: float = 0.2,
        patience: int = 7,
    ):
        """
        Treina a U-Net nos patches do AITEX.

        Fluxo:
          1. Extrai patches de imagens com defeito (stride menor = mais patches)
          2. Extrai patches de imagens normais (máscara toda zero)
          3. Shuffle + split treino/validação
          4. Treina com early stopping (para quando val loss para de melhorar)
        """
        if not TORCH_AVAILABLE:
            print("  ⚠ PyTorch not available. U-Net disabled.")
            return

        print(f"  Device: {self.device}")

        # Extrair patches (stride = patch_size para treino viável em CPU)
        all_patches, all_masks = [], []

        for img, mask in zip(defect_images, defect_masks):
            p, m = self._extract_patches(img, mask, stride=self.patch_size)
            all_patches.extend(p)
            all_masks.extend(m)

        for img in normal_images:
            zero_mask = np.zeros(img.shape[:2], dtype=np.uint8)
            p, m = self._extract_patches(img, zero_mask, stride=self.patch_size * 2)
            all_patches.extend(p)
            all_masks.extend(m)

        n_defect = sum(1 for m in all_masks if m.sum() > 0)
        print(f"  Total patches: {len(all_patches)} ({n_defect} with defects)")

        if len(all_patches) == 0:
            print("  ⚠ No patches extracted. Aborting training.")
            return

        # Limitar patches para treino viável em CPU
        max_patches = 3000
        if len(all_patches) > max_patches:
            # Priorizar patches com defeito
            defect_idx = [i for i, m in enumerate(all_masks) if m.sum() > 0]
            normal_idx = [i for i, m in enumerate(all_masks) if m.sum() == 0]
            np.random.shuffle(normal_idx)
            keep = defect_idx + normal_idx[:max_patches - len(defect_idx)]
            all_patches = [all_patches[i] for i in keep]
            all_masks = [all_masks[i] for i in keep]
            print(f"  Capped to {len(all_patches)} patches for CPU training")

        # Split treino/validação
        indices = np.random.permutation(len(all_patches))
        val_size = max(1, int(len(indices) * val_split))
        val_idx, train_idx = indices[:val_size], indices[val_size:]

        train_ds = FabricPatchDataset(
            [all_patches[i] for i in train_idx],
            [all_masks[i] for i in train_idx], augment=True)
        val_ds = FabricPatchDataset(
            [all_patches[i] for i in val_idx],
            [all_masks[i] for i in val_idx], augment=False)

        train_loader = DataLoader(train_ds, batch_size=batch_size,
                                  shuffle=True, num_workers=0)
        val_loader = DataLoader(val_ds, batch_size=batch_size,
                                shuffle=False, num_workers=0)

        print(f"  Train: {len(train_ds)} | Val: {len(val_ds)}")

        # Setup
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=3, factor=0.5)
        criterion = DiceBCELoss(dice_weight=0.5)

        best_val_loss = float("inf")
        patience_counter = 0
        best_state = None

        # ── Loop de treino ──
        for epoch in range(epochs):
            # Train
            self.model.train()
            train_loss = 0.0
            for batch_img, batch_mask in train_loader:
                batch_img = batch_img.to(self.device)
                batch_mask = batch_mask.to(self.device)

                optimizer.zero_grad()
                output = self.model(batch_img)
                if output.shape != batch_mask.shape:
                    output = F.interpolate(
                        output, size=batch_mask.shape[2:],
                        mode="bilinear", align_corners=True)

                loss = criterion(output, batch_mask)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            train_loss /= len(train_loader)

            # Validate
            self.model.eval()
            val_loss, val_dice = 0.0, 0.0
            with torch.no_grad():
                for batch_img, batch_mask in val_loader:
                    batch_img = batch_img.to(self.device)
                    batch_mask = batch_mask.to(self.device)

                    output = self.model(batch_img)
                    if output.shape != batch_mask.shape:
                        output = F.interpolate(
                            output, size=batch_mask.shape[2:],
                            mode="bilinear", align_corners=True)

                    val_loss += criterion(output, batch_mask).item()
                    pred_bin = (output > 0.5).float()
                    inter = (pred_bin * batch_mask).sum()
                    dice = (2.0 * inter + 1) / (
                        pred_bin.sum() + batch_mask.sum() + 1)
                    val_dice += dice.item()

            val_loss /= len(val_loader)
            val_dice /= len(val_loader)
            scheduler.step(val_loss)

            current_lr = optimizer.param_groups[0]["lr"]
            print(f"  Epoch {epoch+1:3d}/{epochs} | "
                  f"Train: {train_loss:.4f} | "
                  f"Val: {val_loss:.4f} | "
                  f"Dice: {val_dice:.3f} | "
                  f"LR: {current_lr:.1e}")

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_state = {k: v.cpu().clone()
                              for k, v in self.model.state_dict().items()}
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"  Early stopping at epoch {epoch+1}")
                    break

        if best_state is not None:
            self.model.load_state_dict(best_state)
            self.model.to(self.device)

        self.model.eval()
        self._trained = True
        print(f"  ✅ U-Net training complete. Best val loss: {best_val_loss:.4f}")

    # ── Inferência ──

    def predict(
        self, image: np.ndarray,
        threshold: float = 0.5,
        stride: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predição por sliding window com blending de sobreposições.

        Retorna:
          binary_mask:    máscara binária (uint8, 0 ou 255)
          confidence_map: mapa de probabilidade (float32, 0.0 a 1.0)
        """
        if not TORCH_AVAILABLE or not self._trained:
            h, w = image.shape[:2]
            return (np.zeros((h, w), dtype=np.uint8),
                    np.zeros((h, w), dtype=np.float32))

        gray = (cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                if len(image.shape) == 3 else image)
        gray = gray.astype(np.float32) / 255.0

        h, w = gray.shape
        ps = self.patch_size
        stride = stride or ps // 2

        prob_map = np.zeros((h, w), dtype=np.float64)
        count_map = np.zeros((h, w), dtype=np.float64)

        self.model.eval()
        with torch.no_grad():
            for y in range(0, max(1, h - ps + 1), stride):
                for x in range(0, max(1, w - ps + 1), stride):
                    ye, xe = min(y + ps, h), min(x + ps, w)
                    ys, xs = ye - ps, xe - ps
                    patch = gray[ys:ye, xs:xe]
                    if patch.shape != (ps, ps):
                        continue

                    tensor = (torch.FloatTensor(patch)
                              .unsqueeze(0).unsqueeze(0).to(self.device))
                    pred = self.model(tensor).squeeze().cpu().numpy()

                    if pred.shape != (ps, ps):
                        pred = cv2.resize(pred, (ps, ps))

                    prob_map[ys:ye, xs:xe] += pred
                    count_map[ys:ye, xs:xe] += 1.0

        count_map = np.maximum(count_map, 1.0)
        confidence_map = (prob_map / count_map).astype(np.float32)

        binary_mask = (confidence_map > threshold).astype(np.uint8) * 255

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)

        return binary_mask, confidence_map

    # ── Persistência ──

    def save_model(self, path: Path):
        """Salva os pesos do modelo treinado."""
        if TORCH_AVAILABLE and self.model is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(self.model.state_dict(), path)
            print(f"  Model saved: {path}")

    @property
    def is_trained(self) -> bool:
        return self._trained
