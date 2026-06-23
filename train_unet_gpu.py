# -*- coding: utf-8 -*-
"""
Script standalone para treinar a U-Net na GPU e gerar imagens de resultados.
Uso: python train_unet_gpu.py
"""
import sys, os, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
# Pacotes instalados no D:
sys.path.insert(0, r"D:\Andre\Visao\projeto-visao-comp\.packages")
os.environ["MPLBACKEND"] = "Agg"

import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict

# Project imports
from config import (
    AppConfig, RESULTS_DIR, DEFECTS_DIR, DATASET_DIR, MODELS_DIR,
)
from fase2_deteccao.unet_segmentation import UNetSegmenter
from fase1_restauracao.clahe_enhance import CLAHEEnhancer, BrightnessCorrector
from utils.dataset_loader import AitexDataset, DEFECT_TYPE_MAP
from utils.visualization import overlay_heatmap

import torch

OUTPUT_DIR = RESULTS_DIR / "unet_training"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


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


def compute_metrics(pred_mask, gt_mask):
    pred = (pred_mask > 0).astype(bool)
    gt = (gt_mask > 0).astype(bool)
    tp = np.logical_and(pred, gt).sum()
    fp = np.logical_and(pred, ~gt).sum()
    fn = np.logical_and(~pred, gt).sum()
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    intersection = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    iou = intersection / (union + 1e-8)
    return {"iou": float(iou), "f1": float(f1),
            "precision": float(precision), "recall": float(recall)}


def save_training_curves(train_losses, val_losses, val_dices, path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    epochs = range(1, len(train_losses) + 1)

    ax1.plot(epochs, train_losses, "b-o", label="Train Loss", markersize=4)
    ax1.plot(epochs, val_losses, "r-o", label="Val Loss", markersize=4)
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss")
    ax1.set_title("Training & Validation Loss"); ax1.legend(); ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, val_dices, "g-o", label="Val Dice", markersize=4)
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Dice Score")
    ax2.set_title("Validation Dice Score"); ax2.legend(); ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1)

    plt.tight_layout()
    fig.savefig(str(path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def save_example_gallery(examples, path):
    n = len(examples)
    fig, axes = plt.subplots(n, 4, figsize=(20, 5 * n))
    if n == 1:
        axes = axes.reshape(1, -1)

    for i, ex in enumerate(examples):
        # Original
        img_rgb = cv2.cvtColor(ex["original"], cv2.COLOR_BGR2RGB)
        axes[i, 0].imshow(img_rgb)
        axes[i, 0].set_title(f"{ex['defect_type']}\n{ex['filename']}", fontsize=9)
        
        # Ground Truth
        axes[i, 1].imshow(ex["gt_mask"], cmap="Reds")
        axes[i, 1].set_title("Ground Truth Mask")
        
        # U-Net Prediction
        axes[i, 2].imshow(ex["pred_mask"], cmap="Reds")
        m = ex["metrics"]
        axes[i, 2].set_title(f"U-Net Prediction\nIoU={m['iou']:.3f} F1={m['f1']:.3f}")
        
        # Heatmap overlay
        heatmap_rgb = cv2.cvtColor(ex["heatmap"], cv2.COLOR_BGR2RGB)
        axes[i, 3].imshow(heatmap_rgb)
        axes[i, 3].set_title("Confidence Heatmap")

        for j in range(4):
            axes[i, j].axis("off")

    plt.tight_layout()
    fig.savefig(str(path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def save_per_type_chart(type_metrics, path):
    types = list(type_metrics.keys())
    ious = [type_metrics[t]["iou"] for t in types]
    f1s = [type_metrics[t]["f1"] for t in types]

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


class TrackingUNetSegmenter(UNetSegmenter):
    """Extends UNetSegmenter to capture per-epoch metrics."""

    def train(self, defect_images, defect_masks, normal_images,
              epochs=30, lr=1e-3, batch_size=8, val_split=0.2, patience=7):
        from fase2_deteccao.unet_segmentation import (
            FabricPatchDataset, DiceBCELoss,
        )
        from torch.utils.data import DataLoader
        import torch.nn.functional as F
        import torch.optim as optim

        self.history = {"train_loss": [], "val_loss": [], "val_dice": []}

        print(f"  Device: {self.device}")

        all_patches, all_masks = [], []
        for img, mask in zip(defect_images, defect_masks):
            p, m = self._extract_patches(img, mask, stride=self.patch_size)
            all_patches.extend(p); all_masks.extend(m)
        for img in normal_images:
            zero_mask = np.zeros(img.shape[:2], dtype=np.uint8)
            p, m = self._extract_patches(img, zero_mask, stride=self.patch_size * 2)
            all_patches.extend(p); all_masks.extend(m)

        n_defect = sum(1 for m in all_masks if m.sum() > 0)
        print(f"  Total patches: {len(all_patches)} ({n_defect} with defects)")
        if not all_patches:
            return

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

        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=3, factor=0.5)
        criterion = DiceBCELoss(dice_weight=0.5)

        best_val_loss = float("inf")
        patience_counter = 0
        best_state = None

        for epoch in range(epochs):
            self.model.train()
            train_loss = 0.0
            for batch_img, batch_mask in train_loader:
                batch_img = batch_img.to(self.device)
                batch_mask = batch_mask.to(self.device)
                optimizer.zero_grad()
                output = self.model(batch_img)
                if output.shape != batch_mask.shape:
                    output = F.interpolate(output, size=batch_mask.shape[2:],
                                           mode="bilinear", align_corners=True)
                loss = criterion(output, batch_mask)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            train_loss /= len(train_loader)

            self.model.eval()
            val_loss, val_dice = 0.0, 0.0
            with torch.no_grad():
                for batch_img, batch_mask in val_loader:
                    batch_img = batch_img.to(self.device)
                    batch_mask = batch_mask.to(self.device)
                    output = self.model(batch_img)
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
            scheduler.step(val_loss)

            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["val_dice"].append(val_dice)

            lr_now = optimizer.param_groups[0]["lr"]
            print(f"  Epoch {epoch+1:3d}/{epochs} | "
                  f"Train: {train_loss:.4f} | Val: {val_loss:.4f} | "
                  f"Dice: {val_dice:.3f} | LR: {lr_now:.1e}")

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

        if best_state:
            self.model.load_state_dict(best_state)
            self.model.to(self.device)

        self.model.eval()
        self._trained = True
        print(f"  ✅ Training complete. Best val loss: {best_val_loss:.4f}")


def main():
    print("=" * 60)
    print("  U-NET TRAINING SESSION — GPU ACCELERATED")
    print("=" * 60)

    config = AppConfig()
    dl = config.deep_learning

    # Print GPU info
    if torch.cuda.is_available():
        gpu = torch.cuda.get_device_name(0)
        mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  GPU: {gpu} ({mem:.1f} GB)")
    else:
        print("  ⚠ No GPU detected, using CPU")

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
    normal_images = dataset.load_normal_images(max_count=dl.max_normal_images)
    print(f"\n  Defect images: {len(defect_images)}")
    print(f"  Normal images: {len(normal_images)}")

    # Delete old model to force retrain
    model_path = MODELS_DIR / "unet_best.pth"
    if model_path.exists():
        model_path.unlink()
        print("  Deleted old model — will retrain fresh.")

    # Train
    print(f"\n{'=' * 60}")
    print("TRAINING U-NET")
    print("=" * 60)

    unet = TrackingUNetSegmenter(patch_size=dl.unet_patch_size)

    unet.train(
        defect_images=defect_images,
        defect_masks=defect_masks,
        normal_images=normal_images,
        epochs=dl.unet_epochs,
        lr=dl.unet_lr,
        batch_size=dl.unet_batch_size,
        val_split=dl.unet_val_split,
        patience=dl.unet_patience,
    )
    unet.save_model(model_path)

    # Save training curves
    save_training_curves(
        unet.history["train_loss"],
        unet.history["val_loss"],
        unet.history["val_dice"],
        OUTPUT_DIR / "training_curves.png",
    )

    # Inference on all defect images
    print(f"\n{'=' * 60}")
    print("INFERENCE — Generating Results")
    print("=" * 60)

    type_metrics = defaultdict(lambda: {"iou": [], "f1": [], "precision": [], "recall": []})
    all_metrics = []
    examples = []
    seen_types = set()

    for i, (sample, image, mask) in enumerate(dataset.iter_defect_with_masks()):
        preprocessed = preprocess(image, config)
        pred_mask, confidence = unet.predict(preprocessed, dl.unet_threshold)

        # Resize GT if needed
        gt = mask
        if pred_mask.shape[:2] != gt.shape[:2]:
            gt = cv2.resize(gt, (pred_mask.shape[1], pred_mask.shape[0]),
                            interpolation=cv2.INTER_NEAREST)

        m = compute_metrics(pred_mask, gt)
        all_metrics.append(m)
        dt = sample.defect_name
        for k in m:
            type_metrics[dt][k].append(m[k])

        # Collect examples (1 per defect type, max 8)
        if dt not in seen_types and len(examples) < 8:
            seen_types.add(dt)
            heatmap = overlay_heatmap(preprocessed, confidence)
            examples.append({
                "defect_type": dt,
                "filename": sample.image_path.name,
                "original": image,
                "gt_mask": gt,
                "pred_mask": pred_mask,
                "heatmap": heatmap,
                "metrics": m,
            })

        print(f"  [{i+1}/{len(defect_images)}] {sample.image_path.name} — "
              f"{dt} — F1: {m['f1']:.3f} IoU: {m['iou']:.3f}")

    # Aggregate per-type
    type_avg = {}
    for dt, mv in type_metrics.items():
        type_avg[dt] = {k: np.mean(v) for k, v in mv.items()}

    # Save gallery
    if examples:
        save_example_gallery(examples, OUTPUT_DIR / "segmentation_examples.png")

    # Save per-type chart
    if type_avg:
        save_per_type_chart(type_avg, OUTPUT_DIR / "per_defect_performance.png")

    # Summary
    avg_iou = np.mean([m["iou"] for m in all_metrics])
    avg_f1 = np.mean([m["f1"] for m in all_metrics])
    avg_prec = np.mean([m["precision"] for m in all_metrics])
    avg_rec = np.mean([m["recall"] for m in all_metrics])

    print(f"\n{'=' * 60}")
    print("OVERALL RESULTS")
    print("=" * 60)
    print(f"  IoU:       {avg_iou:.4f}")
    print(f"  F1:        {avg_f1:.4f}")
    print(f"  Precision: {avg_prec:.4f}")
    print(f"  Recall:    {avg_rec:.4f}")
    print(f"\n  Results saved to: {OUTPUT_DIR}")
    print(f"  Model saved to:   {model_path}")
    print("✅ Done!")


if __name__ == "__main__":
    main()
