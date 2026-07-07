# -*- coding: utf-8 -*-
"""
Treinamento do Classificador de Defeitos Têxteis

Estágio 2 do pipeline: após a U-Net detectar ONDE está o defeito,
este classificador identifica O QUE é (tipo de defeito).

Etapas:
  1. Extrai patches de defeitos usando as máscaras GT do AITEX
  2. Aplica data augmentation para balancear classes
  3. Treina uma CNN leve (DefectClassifier)
  4. Salva modelo e métricas
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
from collections import Counter
import time
import random

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from config import RESULTS_DIR, DATASET_DIR, MODELS_DIR
from fase2_deteccao.defect_classifier import (
    DefectClassifier, DEFECT_CLASSES, DEFECT_CODE_TO_IDX
)
from utils.dataset_loader import AitexDataset

# ── Configuração ──────────────────────────────────────────────────────────────

OUTPUT_DIR = RESULTS_DIR / "classifier_training"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PATCH_SIZE = 64       # Tamanho do patch para o classificador
EPOCHS = 80
LR = 1e-3
BATCH_SIZE = 32
PATIENCE = 15
AUGMENT_PER_SAMPLE = 20  # Augmentações por patch original

# ── Extração de Patches ──────────────────────────────────────────────────────

def extract_defect_patches(dataset, patch_size=64, padding=16):
    """
    Extrai patches de defeitos do dataset usando máscaras GT.
    Cada patch é centrado no bounding box do defeito e redimensionado.
    
    Retorna: lista de (patch_gray_normalizado, classe_idx)
    """
    patches = []
    
    for sample, image, mask in dataset.iter_defect_with_masks():
        if sample.defect_code not in DEFECT_CODE_TO_IDX:
            continue
        
        class_idx = DEFECT_CODE_TO_IDX[sample.defect_code]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        # Encontrar contornos na máscara GT
        mask_bin = (mask > 0).astype(np.uint8) * 255
        contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 10:  # Ignorar ruído
                continue
            
            x, y, w, h = cv2.boundingRect(cnt)
            
            # Expandir bounding box com padding
            x1 = max(0, x - padding)
            y1 = max(0, y - padding)
            x2 = min(gray.shape[1], x + w + padding)
            y2 = min(gray.shape[0], y + h + padding)
            
            patch = gray[y1:y2, x1:x2]
            if patch.size == 0:
                continue
            
            # Redimensionar para tamanho fixo
            patch_resized = cv2.resize(patch, (patch_size, patch_size),
                                       interpolation=cv2.INTER_LINEAR)
            patch_norm = patch_resized.astype(np.float32) / 255.0
            patches.append((patch_norm, class_idx))
    
    return patches


def augment_patches(patches, augments_per_sample=20):
    """Aplica data augmentation para balancear e expandir o dataset."""
    augmented = []
    
    for patch, label in patches:
        # Manter original
        augmented.append((patch, label))
        
        for _ in range(augments_per_sample):
            aug = patch.copy()
            
            # Flip
            if random.random() > 0.5:
                aug = np.fliplr(aug).copy()
            if random.random() > 0.5:
                aug = np.flipud(aug).copy()
            
            # Rotação 90°
            k = random.randint(0, 3)
            aug = np.rot90(aug, k).copy()
            
            # Brilho
            if random.random() > 0.5:
                aug = np.clip(aug * random.uniform(0.7, 1.3), 0, 1)
            
            # Ruído
            if random.random() > 0.5:
                aug = np.clip(aug + np.random.randn(*aug.shape).astype(np.float32) * 0.03, 0, 1)
            
            # Contraste
            if random.random() > 0.5:
                mean = aug.mean()
                aug = np.clip(random.uniform(0.8, 1.2) * (aug - mean) + mean, 0, 1)
            
            augmented.append((aug.astype(np.float32), label))
    
    return augmented

# ── Dataset PyTorch ──────────────────────────────────────────────────────────

class PatchDataset(Dataset):
    def __init__(self, patches):
        self.patches = patches
    
    def __len__(self):
        return len(self.patches)
    
    def __getitem__(self, idx):
        patch, label = self.patches[idx]
        tensor = torch.FloatTensor(patch).unsqueeze(0)  # (1, 64, 64)
        return tensor, label

# ── Visualização ─────────────────────────────────────────────────────────────

def save_confusion_matrix(y_true, y_pred, path):
    """Salva a matriz de confusão."""
    n = len(DEFECT_CLASSES)
    cm = np.zeros((n, n), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t][p] += 1
    
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    ax.set_title("Matriz de Confusão — Classificador de Defeitos", fontsize=14)
    fig.colorbar(im)
    
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(DEFECT_CLASSES, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(DEFECT_CLASSES, fontsize=9)
    ax.set_xlabel("Predição")
    ax.set_ylabel("Real")
    
    # Números nas células
    for i in range(n):
        for j in range(n):
            color = "white" if cm[i, j] > cm.max() / 2 else "black"
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color=color, fontsize=11)
    
    plt.tight_layout()
    fig.savefig(str(path), dpi=150, bbox_inches="tight")
    plt.close(fig)

def save_training_curves(history, path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    epochs = range(1, len(history["train_loss"]) + 1)
    axes[0].plot(epochs, history["train_loss"], "b-", label="Train")
    axes[0].plot(epochs, history["val_loss"], "r-", label="Val")
    axes[0].set_title("Loss"); axes[0].legend(); axes[0].grid(True, alpha=0.3)
    axes[1].plot(epochs, history["val_acc"], "g-")
    axes[1].set_title("Validation Accuracy"); axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim(0, 1)
    plt.tight_layout()
    fig.savefig(str(path), dpi=150, bbox_inches="tight")
    plt.close(fig)

# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    t0 = time.time()
    print("=" * 60)
    print("  CLASSIFICADOR DE DEFEITOS TÊXTEIS")
    print("  Estágio 2: Identificação do Tipo de Defeito")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    # ── Extrair patches ──
    print("\nEXTRAINDO PATCHES DE DEFEITOS...")
    dataset = AitexDataset(DATASET_DIR)
    raw_patches = extract_defect_patches(dataset, patch_size=PATCH_SIZE)
    
    # Contagem por classe
    class_counts = Counter(label for _, label in raw_patches)
    print(f"  Patches extraídos: {len(raw_patches)}")
    for idx, name in enumerate(DEFECT_CLASSES):
        print(f"    {name:15s}: {class_counts.get(idx, 0)}")

    # ── Augmentar ──
    print(f"\nAUGMENTANDO ({AUGMENT_PER_SAMPLE}× por patch)...")
    all_patches = augment_patches(raw_patches, augments_per_sample=AUGMENT_PER_SAMPLE)
    random.shuffle(all_patches)
    
    aug_counts = Counter(label for _, label in all_patches)
    print(f"  Total patches: {len(all_patches)}")
    for idx, name in enumerate(DEFECT_CLASSES):
        print(f"    {name:15s}: {aug_counts.get(idx, 0)}")

    # ── Split treino/validação ──
    val_size = max(1, int(len(all_patches) * 0.2))
    val_patches = all_patches[:val_size]
    train_patches = all_patches[val_size:]

    train_ds = PatchDataset(train_patches)
    val_ds = PatchDataset(val_patches)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)

    print(f"\n  Train: {len(train_patches)} | Val: {len(val_patches)}")

    # ── Modelo ──
    model = DefectClassifier(num_classes=len(DEFECT_CLASSES)).to(device)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"  Parâmetros: {param_count:,}")

    # Pesos para classes desbalanceadas
    class_weights = []
    total = sum(aug_counts.values())
    for i in range(len(DEFECT_CLASSES)):
        count = aug_counts.get(i, 1)
        class_weights.append(total / (len(DEFECT_CLASSES) * count))
    weights_tensor = torch.FloatTensor(class_weights).to(device)
    
    criterion = nn.CrossEntropyLoss(weight=weights_tensor)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

    # ── Treino ──
    print(f"\nTREINAMENTO ({EPOCHS} epochs)")
    history = {"train_loss": [], "val_loss": [], "val_acc": []}
    best_val_loss = float("inf")
    patience_counter = 0
    best_state = None

    for epoch in range(EPOCHS):
        ep_start = time.time()

        # Train
        model.train()
        train_loss = 0.0
        for batch_img, batch_label in train_loader:
            batch_img = batch_img.to(device, non_blocking=True)
            batch_label = batch_label.to(device, non_blocking=True)
            optimizer.zero_grad()
            output = model(batch_img)
            loss = criterion(output, batch_label)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        # Validate
        model.eval()
        val_loss, correct, total_val = 0.0, 0, 0
        with torch.no_grad():
            for batch_img, batch_label in val_loader:
                batch_img = batch_img.to(device, non_blocking=True)
                batch_label = batch_label.to(device, non_blocking=True)
                output = model(batch_img)
                val_loss += criterion(output, batch_label).item()
                preds = output.argmax(dim=1)
                correct += (preds == batch_label).sum().item()
                total_val += batch_label.size(0)

        val_loss /= len(val_loader)
        val_acc = correct / total_val
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        ep_time = time.time() - ep_start
        marker = " *" if val_loss < best_val_loss else ""
        print(f"  Epoch {epoch+1:3d}/{EPOCHS} | Loss: {train_loss:.4f}/{val_loss:.4f} | Acc: {val_acc:.3f} | {ep_time:.1f}s{marker}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"  Early stopping at epoch {epoch+1}")
                break

    # ── Restaurar melhor modelo ──
    if best_state:
        model.load_state_dict(best_state)
        model.to(device)
    model.eval()

    model_path = MODELS_DIR / "defect_classifier.pth"
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), model_path)
    print(f"\n  Modelo salvo: {model_path}")

    # ── Avaliação final ──
    print("\nAVALIAÇÃO FINAL...")
    all_true, all_pred = [], []
    model.eval()
    with torch.no_grad():
        for batch_img, batch_label in val_loader:
            batch_img = batch_img.to(device)
            output = model(batch_img)
            preds = output.argmax(dim=1)
            all_true.extend(batch_label.numpy())
            all_pred.extend(preds.cpu().numpy())

    # Accuracy por classe
    print(f"\n  Resultados por classe:")
    for idx, name in enumerate(DEFECT_CLASSES):
        mask = np.array(all_true) == idx
        if mask.sum() == 0:
            continue
        acc = (np.array(all_pred)[mask] == idx).mean()
        print(f"    {name:15s}: {acc:.3f} ({mask.sum()} amostras)")

    total_acc = np.mean(np.array(all_true) == np.array(all_pred))
    print(f"\n  Accuracy geral: {total_acc:.4f}")

    # Salvar gráficos
    save_training_curves(history, OUTPUT_DIR / "training_curves.png")
    save_confusion_matrix(all_true, all_pred, OUTPUT_DIR / "confusion_matrix.png")

    print(f"\n  Tempo total: {(time.time() - t0)/60:.1f} min")
    print(f"  Resultados em: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
