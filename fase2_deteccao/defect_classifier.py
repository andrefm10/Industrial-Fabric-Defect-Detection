# -*- coding: utf-8 -*-
"""
Classificador de Tipos de Defeito Têxtil

Arquitetura CNN leve para classificar patches de defeitos detectados
pela U-Net em uma das 7 categorias do dataset AITEX.

Pipeline: U-Net detecta ONDE → Classificador identifica O QUÊ
"""
import torch
import torch.nn as nn


# Mapeamento código → índice (consistente com dataset_loader.py)
DEFECT_CLASSES = [
    "Broken end",    # 0
    "Broken yarn",   # 1
    "Cut",           # 2
    "Fuzzyball",     # 3
    "Pilling",       # 4
    "Nep",           # 5
    "Weft crack",    # 6
]

DEFECT_CODE_TO_IDX = {
    "00": 0, "01": 1, "02": 2, "03": 3,
    "04": 4, "05": 5, "06": 6,
}


class DefectClassifier(nn.Module):
    """
    CNN leve para classificação de patches de defeitos.

    Entrada: patch grayscale 64×64 (1 canal)
    Saída:   logits para 7 classes de defeito

    Arquitetura:
      Conv 1→32 → Conv 32→64 → Conv 64→128 → Conv 128→256
      → Global Average Pooling → Dropout → FC 256→7

    Cada bloco: Conv3×3 → BatchNorm → ReLU → MaxPool2×2
    """
    def __init__(self, num_classes=7, in_channels=1):
        super().__init__()

        self.features = nn.Sequential(
            # Bloco 1: 64×64 → 32×32
            nn.Conv2d(in_channels, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Bloco 2: 32×32 → 16×16
            nn.Conv2d(32, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Bloco 3: 16×16 → 8×8
            nn.Conv2d(64, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Bloco 4: 8×8 → 4×4
            nn.Conv2d(128, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # 4×4 → 1×1
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)

    def predict_class(self, x):
        """Retorna o nome da classe predita para um batch."""
        with torch.no_grad():
            logits = self.forward(x)
            indices = logits.argmax(dim=1)
            return [DEFECT_CLASSES[i] for i in indices.cpu().numpy()]
