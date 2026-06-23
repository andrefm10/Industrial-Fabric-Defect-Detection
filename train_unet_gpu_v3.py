# -*- coding: utf-8 -*-
"""
U-Net GPU Training v3 - Fast Dynamic Sampling & Edge Ignoring
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
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from config import AppConfig, RESULTS_DIR, DATASET_DIR, MODELS_DIR
from fase2_deteccao.unet_segmentation import UNet, DiceBCELoss
from utils.dataset_loader import AitexDataset
from utils.visualization import overlay_heatmap

OUTPUT_DIR = RESULTS_DIR / "unet_training_v3"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PATCH_SIZE = 128
EPOCHS = 50
LR = 3e-4
BATCH_SIZE = 32
PATIENCE = 15
PATCHES_PER_EPOCH = 2000  # Number of random crops per epoch

def get_valid_x_range(image, margin=100):
    """Finds the true fabric area, ignoring padding and conveyor belt edges based on center mean."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    h, w = gray.shape
    
    # Calculate mean of the center of the image (safest part of fabric)
    center_start, center_end = int(w * 0.4), int(w * 0.6)
    center_mean = gray[:, center_start:center_end].mean()
    
    col_means = gray.mean(axis=0)
    # Valid columns are those whose mean is within +- 25 of the center mean
    valid_cols = np.where(np.abs(col_means - center_mean) < 25)[0]
    
    if len(valid_cols) == 0:
        return 0, w
        
    start = max(0, valid_cols[0] + margin)
    end = min(w, valid_cols[-1] - margin)
    
    if start >= end:
        return valid_cols[0], valid_cols[-1]
    return start, end

class DynamicPatchDataset(Dataset):
    def __init__(self, images, masks, patch_size=128, length=2000, augment=True):
        self.images = []
        self.masks = []
        self.valid_ranges = []
        self.defect_boxes = []
        self.patch_size = patch_size
        self.length = length
        self.augment = augment

        for img, mask in zip(images, masks):
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
            gray = gray.astype(np.float32) / 255.0
            mask_norm = (mask > 0).astype(np.float32)
            
            self.images.append(gray)
            self.masks.append(mask_norm)
            self.valid_ranges.append(get_valid_x_range(img))
            
            # Find bounding boxes of defects for targeted sampling
            boxes = []
            if mask_norm.sum() > 0:
                contours, _ = cv2.findContours((mask_norm * 255).astype(np.uint8), 
                                               cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for cnt in contours:
                    x, y, w, h = cv2.boundingRect(cnt)
                    boxes.append((x, y, w, h))
            self.defect_boxes.append(boxes)

    def __len__(self):
        return self.length

    def _augment(self, image, mask):
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
            factor = np.random.uniform(0.7, 1.3)
            image = np.clip(image * factor, 0, 1)
        if np.random.rand() > 0.5:
            noise = np.random.randn(*image.shape).astype(np.float32) * 0.03
            image = np.clip(image + noise, 0, 1)
        if np.random.rand() > 0.5:
            alpha = np.random.uniform(0.8, 1.2)
            mean = image.mean()
            image = np.clip(alpha * (image - mean) + mean, 0, 1)
        return image.astype(np.float32), mask.astype(np.float32)

    def __getitem__(self, idx):
        # Pick a random image
        img_idx = random.randint(0, len(self.images) - 1)
        img = self.images[img_idx]
        mask = self.masks[img_idx]
        x_start_valid, x_end_valid = self.valid_ranges[img_idx]
        boxes = self.defect_boxes[img_idx]

        h, w = img.shape
        ps = self.patch_size

        # 50% chance to sample around a defect (if any exist in this image)
        if boxes and random.random() < 0.5:
            bx, by, bw, bh = random.choice(boxes)
            # Pick a crop that covers at least part of the bounding box
            min_x = max(x_start_valid, bx - ps + 1)
            max_x = min(x_end_valid - ps, bx + bw - 1)
            if max_x < min_x: max_x = min_x # Fallback
            
            min_y = max(0, by - ps + 1)
            max_y = min(h - ps, by + bh - 1)
            if max_y < min_y: max_y = min_y
            
            x = random.randint(int(min_x), int(max_x))
            y = random.randint(int(min_y), int(max_y))
        else:
            # Random crop from valid region
            max_x = max(x_start_valid, x_end_valid - ps)
            x = random.randint(int(x_start_valid), int(max_x))
            y = random.randint(0, h - ps)

        # Safety bounds
        x = np.clip(x, 0, w - ps)
        y = np.clip(y, 0, h - ps)

        patch_img = img[y:y+ps, x:x+ps]
        patch_mask = mask[y:y+ps, x:x+ps]

        if self.augment:
            patch_img, patch_mask = self._augment(patch_img, patch_mask)

        return (torch.FloatTensor(patch_img).unsqueeze(0),
                torch.FloatTensor(patch_mask).unsqueeze(0))

def predict_image(model, image, device, patch_size=128, stride=64, threshold=0.5):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    gray = gray.astype(np.float32) / 255.0
    h, w = gray.shape
    ps = patch_size
    prob_map = np.zeros((h, w), dtype=np.float64)
    count_map = np.zeros((h, w), dtype=np.float64)

    # Only predict in valid region
    x_start_valid, x_end_valid = get_valid_x_range(image)

    model.eval()
    with torch.no_grad():
        for y in range(0, max(1, h - ps + 1), stride):
            for x in range(x_start_valid, max(x_start_valid + 1, x_end_valid - ps + 1), stride):
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
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
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
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    epochs = range(1, len(history["train_loss"]) + 1)
    axes[0].plot(epochs, history["train_loss"], "b-", label="Train")
    axes[0].plot(epochs, history["val_loss"], "r-", label="Val")
    axes[0].set_title("Loss Curves"); axes[0].legend(); axes[0].grid(True, alpha=0.3)
    axes[1].plot(epochs, history["val_dice"], "g-")
    axes[1].set_title("Validation Dice"); axes[1].grid(True, alpha=0.3); axes[1].set_ylim(0, 1)
    plt.tight_layout()
    fig.savefig(str(path), dpi=150, bbox_inches="tight")
    plt.close(fig)

def save_gallery(examples, path):
    n = len(examples)
    fig, axes = plt.subplots(n, 4, figsize=(24, 4 * n))
    if n == 1: axes = axes.reshape(1, -1)
    for i, ex in enumerate(examples):
        img_rgb = cv2.cvtColor(ex["original"], cv2.COLOR_BGR2RGB)
        axes[i, 0].imshow(img_rgb); axes[i, 0].set_title(f"{ex['type']}\n{ex['name']}")
        axes[i, 1].imshow(ex["gt"], cmap="Reds"); axes[i, 1].set_title("Ground Truth")
        axes[i, 2].imshow(ex["pred"], cmap="Reds")
        axes[i, 2].set_title(f"Prediction\nIoU={ex['metrics']['iou']:.3f} F1={ex['metrics']['f1']:.3f}")
        axes[i, 3].imshow(cv2.cvtColor(ex["heatmap"], cv2.COLOR_BGR2RGB)); axes[i, 3].set_title("Heatmap")
        for j in range(4): axes[i, j].axis("off")
    plt.tight_layout()
    fig.savefig(str(path), dpi=150, bbox_inches="tight")
    plt.close(fig)

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

def main():
    t0 = time.time()
    print("=" * 60)
    print("  U-NET TRAINING v3 - FAST DYNAMIC & EDGE IGNORING")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    # Load dataset directly into memory
    print("\nLOADING IMAGES INTO MEMORY...")
    dataset = AitexDataset(DATASET_DIR)
    
    all_images = []
    all_masks = []
    for sample, image, mask in dataset.iter_defect_with_masks():
        all_images.append(image)
        all_masks.append(mask)
    
    # Also load some normal images
    normal_images = dataset.load_normal_images(max_count=20)
    for img in normal_images:
        all_images.append(img)
        all_masks.append(np.zeros(img.shape[:2], dtype=np.uint8))
        
    print(f"  Total images loaded: {len(all_images)}")

    # Split
    indices = np.random.permutation(len(all_images))
    val_size = max(1, int(len(indices) * 0.15))
    val_idx, train_idx = indices[:val_size], indices[val_size:]

    train_ds = DynamicPatchDataset(
        [all_images[i] for i in train_idx],
        [all_masks[i] for i in train_idx],
        length=PATCHES_PER_EPOCH, augment=True
    )
    val_ds = DynamicPatchDataset(
        [all_images[i] for i in val_idx],
        [all_masks[i] for i in val_idx],
        length=PATCHES_PER_EPOCH // 4, augment=False
    )

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)

    print(f"  Train patches/epoch: {PATCHES_PER_EPOCH}")
    print(f"  Val patches/epoch:   {PATCHES_PER_EPOCH // 4}")

    model = UNet(in_channels=1, out_channels=1, features=[32, 64, 128, 256]).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
    criterion = DiceBCELoss(dice_weight=0.7)

    # Delete old model
    model_path = MODELS_DIR / "unet_best.pth"
    if model_path.exists(): model_path.unlink()

    print(f"\nTRAINING ({EPOCHS} epochs, batch={BATCH_SIZE}, lr={LR})")
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
                output = F.interpolate(output, size=batch_mask.shape[2:], mode="bilinear")
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
                    output = F.interpolate(output, size=batch_mask.shape[2:], mode="bilinear")
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
        print(f"  Epoch {epoch+1:3d}/{EPOCHS} | Train: {train_loss:.4f} | Val: {val_loss:.4f} | Dice: {val_dice:.3f} | {ep_time:.1f}s")

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

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), model_path)
    save_training_curves(history, OUTPUT_DIR / "training_curves.png")

    print("\nINFERENCE...")
    type_metrics = defaultdict(lambda: {"iou": [], "f1": [], "precision": [], "recall": []})
    all_metrics = []
    examples = []
    seen_types = set()

    # Inference using original raw image reading loop to get sample names
    for i, (sample, image, mask) in enumerate(dataset.iter_defect_with_masks()):
        pred_mask, confidence = predict_image(model, image, device, PATCH_SIZE, stride=64, threshold=0.5)
        
        gt = mask
        if pred_mask.shape[:2] != gt.shape[:2]:
            gt = cv2.resize(gt, (pred_mask.shape[1], pred_mask.shape[0]), interpolation=cv2.INTER_NEAREST)
        
        m = compute_metrics(pred_mask, gt)
        all_metrics.append(m)
        dt = sample.defect_name
        for k in m: type_metrics[dt][k].append(m[k])

        if dt not in seen_types and len(examples) < 8:
            seen_types.add(dt)
            heatmap = overlay_heatmap(image, confidence)
            examples.append({
                "type": dt, "name": sample.image_path.name,
                "original": image, "gt": gt, "pred": pred_mask,
                "heatmap": heatmap, "metrics": m,
            })
        if (i + 1) % 20 == 0 or i == 0:
            print(f"  [{i+1}/106] {sample.image_path.name} - {dt} - F1: {m['f1']:.3f}")

    type_avg = {dt: {k: np.mean(v) for k, v in mv.items()} for dt, mv in type_metrics.items()}
    if examples: save_gallery(examples, OUTPUT_DIR / "segmentation_examples.png")
    if type_avg: save_type_chart(type_avg, OUTPUT_DIR / "per_defect_performance.png")

    avg_m = {k: np.mean([m[k] for m in all_metrics]) for k in ["iou", "f1", "precision", "recall"]}
    print(f"\nRESULTS: IoU={avg_m['iou']:.4f} F1={avg_m['f1']:.4f}")
    print(f"Time: {(time.time() - t0)/60:.1f} min. Done!")

if __name__ == "__main__":
    main()
