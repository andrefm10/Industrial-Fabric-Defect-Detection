# -*- coding: utf-8 -*-
"""
U-Net GPU Training v2 - Treino melhorado com mais epochs e oversampling.
"""
import sys, os, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, r"D:\Andre\Visao\projeto-visao-comp\.packages")
os.environ["MPLBACKEND"] = "Agg"

import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from config import AppConfig, RESULTS_DIR, DATASET_DIR, MODELS_DIR
from fase2_deteccao.unet_segmentation import UNet, DiceBCELoss
from fase1_restauracao.clahe_enhance import CLAHEEnhancer, BrightnessCorrector
from utils.dataset_loader import AitexDataset
from utils.visualization import overlay_heatmap

OUTPUT_DIR = RESULTS_DIR / "unet_training_v2"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PATCH_SIZE = 128
EPOCHS = 80
LR = 3e-4
BATCH_SIZE = 32
PATIENCE = 15


class AugmentedDataset(Dataset):
    """Dataset com augmentation mais agressivo e oversampling de defeitos."""
    def __init__(self, patches, masks, augment=True, oversample_defect=3):
        self.augment = augment
        # Oversample patches com defeito
        self.patches = list(patches)
        self.masks = list(masks)
        if oversample_defect > 1:
            defect_p = [(p, m) for p, m in zip(patches, masks) if m.sum() > 0]
            for _ in range(oversample_defect - 1):
                for p, m in defect_p:
                    self.patches.append(p)
                    self.masks.append(m)

    def __len__(self):
        return len(self.patches)

    def _augment(self, image, mask):
        # Random flip H
        if np.random.rand() > 0.5:
            image = np.fliplr(image).copy()
            mask = np.fliplr(mask).copy()
        # Random flip V
        if np.random.rand() > 0.5:
            image = np.flipud(image).copy()
            mask = np.flipud(mask).copy()
        # Random rotation 90
        k = np.random.randint(0, 4)
        image = np.rot90(image, k).copy()
        mask = np.rot90(mask, k).copy()
        # Brightness
        if np.random.rand() > 0.5:
            factor = np.random.uniform(0.7, 1.3)
            image = np.clip(image * factor, 0, 1)
        # Noise
        if np.random.rand() > 0.5:
            noise = np.random.randn(*image.shape).astype(np.float32) * 0.03
            image = np.clip(image + noise, 0, 1)
        # Contrast
        if np.random.rand() > 0.5:
            alpha = np.random.uniform(0.8, 1.2)
            mean = image.mean()
            image = np.clip(alpha * (image - mean) + mean, 0, 1)
        return image.astype(np.float32), mask.astype(np.float32)

    def __getitem__(self, idx):
        image = self.patches[idx].astype(np.float32)
        mask = self.masks[idx].astype(np.float32)
        if self.augment:
            image, mask = self._augment(image, mask)
        return (torch.FloatTensor(image).unsqueeze(0),
                torch.FloatTensor(mask).unsqueeze(0))


def extract_patches(image, mask, patch_size, stride):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    gray = gray.astype(np.float32) / 255.0
    mask_norm = (mask > 0).astype(np.float32)
    h, w = gray.shape
    ps = patch_size
    patches, mask_patches = [], []
    for y in range(0, max(1, h - ps + 1), stride):
        for x in range(0, max(1, w - ps + 1), stride):
            ye, xe = min(y + ps, h), min(x + ps, w)
            ys, xs = ye - ps, xe - ps
            p = gray[ys:ye, xs:xe]
            m = mask_norm[ys:ye, xs:xe]
            if p.shape == (ps, ps):
                patches.append(p)
                mask_patches.append(m)
    return patches, mask_patches


def preprocess(image, config):
    bc = BrightnessCorrector(
        target_mean=config.detection.brightness_target_mean,
        tolerance=config.detection.brightness_tolerance,
    )
    image = bc.auto_correct(image)
    enh = CLAHEEnhancer(
        clip_limit=config.restoration.clahe_clip_limit,
        grid_size=config.restoration.clahe_grid_size,
        sharpening_strength=config.restoration.sharpening_strength,
    )
    return enh.enhance(image)


def predict_image(model, image, device, patch_size=128, stride=64, threshold=0.5):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    gray = gray.astype(np.float32) / 255.0
    h, w = gray.shape
    ps = patch_size
    prob_map = np.zeros((h, w), dtype=np.float64)
    count_map = np.zeros((h, w), dtype=np.float64)

    model.eval()
    with torch.no_grad():
        for y in range(0, max(1, h - ps + 1), stride):
            for x in range(0, max(1, w - ps + 1), stride):
                ye, xe = min(y + ps, h), min(x + ps, w)
                ys, xs = ye - ps, xe - ps
                patch = gray[ys:ye, xs:xe]
                if patch.shape != (ps, ps):
                    continue
                tensor = torch.FloatTensor(patch).unsqueeze(0).unsqueeze(0).to(device)
                pred = model(tensor).squeeze().cpu().numpy()
                if pred.shape != (ps, ps):
                    pred = cv2.resize(pred, (ps, ps))
                prob_map[ys:ye, xs:xe] += pred
                count_map[ys:ye, xs:xe] += 1.0

    count_map = np.maximum(count_map, 1.0)
    confidence = (prob_map / count_map).astype(np.float32)
    binary = (confidence > threshold).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    return binary, confidence


def compute_metrics(pred_mask, gt_mask):
    pred = (pred_mask > 0).astype(bool)
    gt = (gt_mask > 0).astype(bool)
    tp = np.logical_and(pred, gt).sum()
    fp = np.logical_and(pred, ~gt).sum()
    fn = np.logical_and(~pred, gt).sum()
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    iou = np.logical_and(pred, gt).sum() / (np.logical_or(pred, gt).sum() + 1e-8)
    return {"iou": float(iou), "f1": float(f1),
            "precision": float(precision), "recall": float(recall)}


def save_training_curves(history, path):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    epochs = range(1, len(history["train_loss"]) + 1)

    axes[0].plot(epochs, history["train_loss"], "b-", label="Train", linewidth=2)
    axes[0].plot(epochs, history["val_loss"], "r-", label="Val", linewidth=2)
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss Curves"); axes[0].legend(); axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, history["val_dice"], "g-", linewidth=2)
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Dice")
    axes[1].set_title("Validation Dice"); axes[1].grid(True, alpha=0.3); axes[1].set_ylim(0, 1)

    axes[2].plot(epochs, history["lr"], "m-", linewidth=2)
    axes[2].set_xlabel("Epoch"); axes[2].set_ylabel("Learning Rate")
    axes[2].set_title("Learning Rate Schedule"); axes[2].grid(True, alpha=0.3)
    axes[2].set_yscale("log")

    plt.tight_layout()
    fig.savefig(str(path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def save_gallery(examples, path):
    n = len(examples)
    fig, axes = plt.subplots(n, 4, figsize=(24, 4 * n))
    if n == 1:
        axes = axes.reshape(1, -1)
    for i, ex in enumerate(examples):
        img_rgb = cv2.cvtColor(ex["original"], cv2.COLOR_BGR2RGB)
        axes[i, 0].imshow(img_rgb); axes[i, 0].set_title(f"{ex['type']}\n{ex['name']}", fontsize=9)
        axes[i, 1].imshow(ex["gt"], cmap="Reds"); axes[i, 1].set_title("Ground Truth")
        axes[i, 2].imshow(ex["pred"], cmap="Reds")
        m = ex["metrics"]
        axes[i, 2].set_title(f"Prediction\nIoU={m['iou']:.3f} F1={m['f1']:.3f}")
        hm_rgb = cv2.cvtColor(ex["heatmap"], cv2.COLOR_BGR2RGB)
        axes[i, 3].imshow(hm_rgb); axes[i, 3].set_title("Heatmap")
        for j in range(4):
            axes[i, j].axis("off")
    plt.tight_layout()
    fig.savefig(str(path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def save_type_chart(type_avg, path):
    types = sorted(type_avg.keys())
    ious = [type_avg[t]["iou"] for t in types]
    f1s = [type_avg[t]["f1"] for t in types]
    x = np.arange(len(types))
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - 0.2, ious, 0.35, label="IoU", color="#4c72b0")
    ax.bar(x + 0.2, f1s, 0.35, label="F1", color="#dd8452")
    ax.set_xticks(x); ax.set_xticklabels(types, rotation=30, ha="right")
    ax.set_ylabel("Score"); ax.set_title("U-Net Performance by Defect Type")
    ax.legend(); ax.grid(axis="y", alpha=0.3); ax.set_ylim(0, 1)
    plt.tight_layout()
    fig.savefig(str(path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def main():
    t0 = time.time()
    print("=" * 60)
    print("  U-NET TRAINING v2 - GPU ACCELERATED")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        gpu = torch.cuda.get_device_name(0)
        mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  GPU: {gpu} ({mem:.1f} GB)")
    print(f"  Device: {device}")

    config = AppConfig()

    # Load dataset
    print(f"\n{'=' * 60}")
    print("LOADING DATASET")
    print("=" * 60)
    dataset = AitexDataset(DATASET_DIR)
    print(dataset.summary())

    defect_images, defect_masks = [], []
    for sample, image, mask in dataset.iter_defect_with_masks():
        defect_images.append(image)
        defect_masks.append(mask)
    normal_images = dataset.load_normal_images(max_count=20)

    # Extract patches - defect images with small stride for more defect coverage
    print(f"\n{'=' * 60}")
    print("EXTRACTING PATCHES")
    print("=" * 60)
    all_patches, all_masks = [], []

    for img, mask in zip(defect_images, defect_masks):
        # Small stride for defect images to capture more defect patches
        p, m = extract_patches(img, mask, PATCH_SIZE, stride=PATCH_SIZE // 2)
        all_patches.extend(p)
        all_masks.extend(m)

    for img in normal_images:
        zero_mask = np.zeros(img.shape[:2], dtype=np.uint8)
        p, m = extract_patches(img, zero_mask, PATCH_SIZE, stride=PATCH_SIZE * 2)
        all_patches.extend(p)
        all_masks.extend(m)

    n_defect = sum(1 for m in all_masks if m.sum() > 0)
    print(f"  Total patches: {len(all_patches)} ({n_defect} with defects)")

    # Split
    indices = np.random.permutation(len(all_patches))
    val_size = max(1, int(len(indices) * 0.15))
    val_idx, train_idx = indices[:val_size], indices[val_size:]

    train_ds = AugmentedDataset(
        [all_patches[i] for i in train_idx],
        [all_masks[i] for i in train_idx],
        augment=True, oversample_defect=5,
    )
    val_ds = AugmentedDataset(
        [all_patches[i] for i in val_idx],
        [all_masks[i] for i in val_idx],
        augment=False, oversample_defect=1,
    )

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=0, pin_memory=True)

    print(f"  Train: {len(train_ds)} (with 5x defect oversampling)")
    print(f"  Val:   {len(val_ds)}")

    # Model
    model = UNet(in_channels=1, out_channels=1, features=[32, 64, 128, 256]).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model params: {n_params:,}")

    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
    criterion = DiceBCELoss(dice_weight=0.7)  # Higher dice weight for imbalanced data

    # Delete old model
    model_path = MODELS_DIR / "unet_best.pth"
    if model_path.exists():
        model_path.unlink()

    # Training
    print(f"\n{'=' * 60}")
    print(f"TRAINING ({EPOCHS} epochs, batch={BATCH_SIZE}, lr={LR})")
    print("=" * 60)

    history = {"train_loss": [], "val_loss": [], "val_dice": [], "lr": []}
    best_val_loss = float("inf")
    patience_counter = 0
    best_state = None

    for epoch in range(EPOCHS):
        ep_start = time.time()
        model.train()
        train_loss = 0.0
        for batch_img, batch_mask in train_loader:
            batch_img = batch_img.to(device, non_blocking=True)
            batch_mask = batch_mask.to(device, non_blocking=True)
            optimizer.zero_grad()
            output = model(batch_img)
            if output.shape != batch_mask.shape:
                output = F.interpolate(output, size=batch_mask.shape[2:],
                                       mode="bilinear", align_corners=True)
            loss = criterion(output, batch_mask)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        model.eval()
        val_loss, val_dice = 0.0, 0.0
        with torch.no_grad():
            for batch_img, batch_mask in val_loader:
                batch_img = batch_img.to(device, non_blocking=True)
                batch_mask = batch_mask.to(device, non_blocking=True)
                output = model(batch_img)
                if output.shape != batch_mask.shape:
                    output = F.interpolate(output, size=batch_mask.shape[2:],
                                           mode="bilinear", align_corners=True)
                val_loss += criterion(output, batch_mask).item()
                pred_bin = (output > 0.5).float()
                inter = (pred_bin * batch_mask).sum()
                dice = (2.0 * inter + 1) / (pred_bin.sum() + batch_mask.sum() + 1)
                val_dice += dice.item()

        val_loss /= len(val_loader)
        val_dice /= len(val_loader)

        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_dice"].append(val_dice)
        history["lr"].append(current_lr)

        ep_time = time.time() - ep_start
        print(f"  Epoch {epoch+1:3d}/{EPOCHS} | "
              f"Train: {train_loss:.4f} | Val: {val_loss:.4f} | "
              f"Dice: {val_dice:.3f} | LR: {current_lr:.1e} | {ep_time:.1f}s")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"  Early stopping at epoch {epoch+1}")
                break

    if best_state:
        model.load_state_dict(best_state)
        model.to(device)
    model.eval()

    # Save model
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), model_path)
    print(f"  Model saved: {model_path}")
    print(f"  Best val loss: {best_val_loss:.4f}")

    # Save training curves
    save_training_curves(history, OUTPUT_DIR / "training_curves.png")

    # Inference
    print(f"\n{'=' * 60}")
    print("INFERENCE")
    print("=" * 60)

    type_metrics = defaultdict(lambda: {"iou": [], "f1": [], "precision": [], "recall": []})
    all_metrics = []
    examples = []
    seen_types = set()

    for i, (sample, image, mask) in enumerate(dataset.iter_defect_with_masks()):
        preprocessed = preprocess(image, config)
        pred_mask, confidence = predict_image(model, preprocessed, device,
                                              PATCH_SIZE, stride=64, threshold=0.5)
        gt = mask
        if pred_mask.shape[:2] != gt.shape[:2]:
            gt = cv2.resize(gt, (pred_mask.shape[1], pred_mask.shape[0]),
                            interpolation=cv2.INTER_NEAREST)
        m = compute_metrics(pred_mask, gt)
        all_metrics.append(m)
        dt = sample.defect_name
        for k in m:
            type_metrics[dt][k].append(m[k])

        if dt not in seen_types and len(examples) < 8:
            seen_types.add(dt)
            heatmap = overlay_heatmap(preprocessed, confidence)
            examples.append({
                "type": dt, "name": sample.image_path.name,
                "original": image, "gt": gt, "pred": pred_mask,
                "heatmap": heatmap, "metrics": m,
            })
        if (i + 1) % 20 == 0 or i == 0:
            print(f"  [{i+1}/{len(defect_images)}] {sample.image_path.name} - "
                  f"{dt} - F1: {m['f1']:.3f} IoU: {m['iou']:.3f}")

    type_avg = {dt: {k: np.mean(v) for k, v in mv.items()} for dt, mv in type_metrics.items()}

    if examples:
        save_gallery(examples, OUTPUT_DIR / "segmentation_examples.png")
    if type_avg:
        save_type_chart(type_avg, OUTPUT_DIR / "per_defect_performance.png")

    # Summary
    avg_m = {k: np.mean([m[k] for m in all_metrics]) for k in ["iou", "f1", "precision", "recall"]}
    elapsed = time.time() - t0

    print(f"\n{'=' * 60}")
    print("RESULTS")
    print("=" * 60)
    print(f"  IoU:       {avg_m['iou']:.4f}")
    print(f"  F1:        {avg_m['f1']:.4f}")
    print(f"  Precision: {avg_m['precision']:.4f}")
    print(f"  Recall:    {avg_m['recall']:.4f}")
    print(f"  Total time: {elapsed/60:.1f} min")
    print(f"  Output: {OUTPUT_DIR}")
    print("[DONE]")


if __name__ == "__main__":
    main()
