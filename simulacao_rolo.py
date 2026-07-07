# -*- coding: utf-8 -*-
"""
Simulação de Rolo Industrial — Inspeção Têxtil em Tempo Real com U-Net

Simula a passagem contínua de tecido por uma câmera de inspeção industrial.
As imagens do dataset AITEX são concatenadas horizontalmente formando um
"rolo" longo, que desliza da direita para a esquerda na tela.

A U-Net processa cada frame visível e sobrepõe um overlay vermelho
nos defeitos detectados, junto com bounding boxes e rótulos.

Controles:
  ESPAÇO  — Pausar / Retomar
  +/-     — Aumentar / Diminuir velocidade da esteira
  T       — Alternar entre exibição com/sem heatmap
  Q / ESC — Sair
  S       — Salvar frame atual como imagem
"""
import sys
import os
sys.path.insert(0, r"D:\Andre\Visao\projeto-visao-comp\.packages")

import numpy as np
import cv2
import time
from pathlib import Path

import torch

from config import DATASET_DIR, MODELS_DIR, RESULTS_DIR
from fase2_deteccao.unet_segmentation import UNet
from fase2_deteccao.defect_classifier import DefectClassifier, DEFECT_CLASSES
from utils.dataset_loader import AitexDataset

# ── Configuração ──────────────────────────────────────────────────────────────

WINDOW_WIDTH = 1280       # Largura da janela de visualização (pixels)
WINDOW_HEIGHT = 256       # Altura da janela (altura natural das imagens AITEX)
SCROLL_SPEED = 4          # Pixels por frame de deslocamento do rolo
MIN_SPEED = 1
MAX_SPEED = 30
PATCH_SIZE = 128           # Deve coincidir com o treinamento
INFERENCE_STRIDE = 64      # Stride da sliding window
THRESHOLD = 0.5            # Limiar de confiança para detecção
FPS_TARGET = 30            # FPS alvo da simulação

OUTPUT_DIR = RESULTS_DIR / "simulacao_rolo"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ── Funções auxiliares ────────────────────────────────────────────────────────


def get_valid_x_range(image, margin=150):
    """Encontra a faixa X do tecido válido, ignorando bordas de esteira."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    h, w = gray.shape
    center_start, center_end = int(w * 0.4), int(w * 0.6)
    center_mean = gray[:, center_start:center_end].mean()
    col_means = gray.mean(axis=0)
    valid_cols = np.where(np.abs(col_means - center_mean) < 25)[0]
    if len(valid_cols) == 0:
        return 0, w
    start = max(0, valid_cols[0] + margin)
    end = min(w, valid_cols[-1] - margin)
    if start >= end:
        return valid_cols[0], valid_cols[-1]
    return start, end


def crop_to_valid_region(image, mask=None):
    """Recorta a imagem para conter apenas o tecido válido."""
    x_start, x_end = get_valid_x_range(image)
    cropped_img = image[:, x_start:x_end]
    if mask is not None:
        cropped_mask = mask[:, x_start:x_end]
        return cropped_img, cropped_mask
    return cropped_img, None


def predict_region(model, region_gray, device, patch_size=128, stride=64, threshold=0.5):
    """
    Faz inferência U-Net numa região de tecido usando sliding window.
    Retorna a máscara binária e o mapa de confiança.
    """
    h, w = region_gray.shape
    ps = patch_size
    prob_map = np.zeros((h, w), dtype=np.float64)
    count_map = np.zeros((h, w), dtype=np.float64)

    model.eval()
    with torch.no_grad():
        for y in range(0, max(1, h - ps + 1), stride):
            for x in range(0, max(1, w - ps + 1), stride):
                ye, xe = min(y + ps, h), min(x + ps, w)
                ys, xs = ye - ps, xe - ps
                patch = region_gray[ys:ye, xs:xe]
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


def blend_transition(img_left, img_right, blend_width=60):
    """
    Cria uma transição suave (feathering) entre duas imagens adjacentes.
    Em vez de uma costura brusca, os últimos `blend_width` pixels da esquerda
    são interpolados gradualmente com os primeiros `blend_width` da direita.
    Retorna as duas imagens modificadas (in-place safe).
    """
    h = img_left.shape[0]
    bw = min(blend_width, img_left.shape[1] // 4, img_right.shape[1] // 4)
    if bw < 2:
        return img_left, img_right

    # Criar gradiente linear: 1.0 → 0.0 (esquerda some) e 0.0 → 1.0 (direita aparece)
    alpha = np.linspace(1.0, 0.0, bw).reshape(1, bw, 1).astype(np.float32)
    alpha = np.broadcast_to(alpha, (h, bw, 3 if len(img_left.shape) == 3 else 1))

    left_region = img_left[:, -bw:].astype(np.float32)
    right_region = img_right[:, :bw].astype(np.float32)

    blended = (left_region * alpha + right_region * (1.0 - alpha)).astype(np.uint8)

    img_left = img_left.copy()
    img_right = img_right.copy()
    img_left[:, -bw:] = blended
    img_right[:, :bw] = blended

    return img_left, img_right


def build_fabric_roll(dataset):
    """
    Monta o rolo contínuo de tecido concatenando imagens do dataset.
    Aplica feathering nas junções para suavizar transições entre segmentos.
    Retorna: (rolo_bgr, rolo_mask, lista_de_segmentos)
    """
    segments = []
    roll_parts_img = []
    roll_parts_mask = []
    current_x = 0

    # Adicionar imagens com defeito
    for sample, image, mask in dataset.iter_defect_with_masks():
        cropped_img, cropped_mask = crop_to_valid_region(image, mask)
        h, w = cropped_img.shape[:2]
        if w < 100:
            continue

        segments.append({
            "name": sample.image_path.name,
            "defect_type": sample.defect_name,
            "x_start": current_x,
            "x_end": current_x + w,
            "has_defect": True,
        })
        roll_parts_img.append(cropped_img)
        roll_parts_mask.append(cropped_mask)
        current_x += w

    # Adicionar algumas imagens normais entre as com defeito
    normal_images = dataset.load_normal_images(max_count=15)
    for i, img in enumerate(normal_images):
        cropped_img, _ = crop_to_valid_region(img)
        h, w = cropped_img.shape[:2]
        if w < 100:
            continue

        segments.append({
            "name": f"normal_{i:03d}",
            "defect_type": "Normal",
            "x_start": current_x,
            "x_end": current_x + w,
            "has_defect": False,
        })
        roll_parts_img.append(cropped_img)
        roll_parts_mask.append(np.zeros((h, w), dtype=np.uint8))
        current_x += w

    # Embaralhar os segmentos para misturar normal e defeito
    indices = np.random.permutation(len(segments))
    shuffled_img = [roll_parts_img[i] for i in indices]
    shuffled_mask = [roll_parts_mask[i] for i in indices]
    shuffled_segments = [segments[i] for i in indices]

    # Aplicar feathering nas junções entre segmentos adjacentes
    for i in range(len(shuffled_img) - 1):
        shuffled_img[i], shuffled_img[i + 1] = blend_transition(
            shuffled_img[i], shuffled_img[i + 1], blend_width=60
        )

    # Recalcular posições após embaralhar
    current_x = 0
    for i, seg in enumerate(shuffled_segments):
        w = shuffled_img[i].shape[1]
        seg["x_start"] = current_x
        seg["x_end"] = current_x + w
        current_x += w

    roll_img = np.concatenate(shuffled_img, axis=1)
    roll_mask = np.concatenate(shuffled_mask, axis=1)

    return roll_img, roll_mask, shuffled_segments


def draw_hud(frame, scroll_x, roll_width, speed, paused, show_heatmap,
             current_segment, defects_found, fps):
    """Desenha o HUD (heads-up display) informativo sobre o frame."""
    h, w = frame.shape[:2]

    # Barra superior semi-transparente
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 50), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    # Barra inferior semi-transparente
    overlay2 = frame.copy()
    cv2.rectangle(overlay2, (0, h - 40), (w, h), (20, 20, 20), -1)
    cv2.addWeighted(overlay2, 0.7, frame, 0.3, 0, frame)

    # Título
    cv2.putText(frame, "SIMULACAO ROLO INDUSTRIAL - INSPECAO TEXTIL",
                (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

    # Info do segmento atual
    seg_text = f"Tecido: {current_segment['name']}  |  Tipo: {current_segment['defect_type']}"
    cv2.putText(frame, seg_text,
                (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

    # Barra de progresso
    progress = scroll_x / max(1, roll_width - w)
    bar_y = h - 30
    bar_w = w - 20
    cv2.rectangle(frame, (10, bar_y), (10 + bar_w, bar_y + 10), (60, 60, 60), -1)
    cv2.rectangle(frame, (10, bar_y), (10 + int(bar_w * progress), bar_y + 10), (0, 200, 100), -1)

    # Stats
    status = "PAUSADO" if paused else "RODANDO"
    status_color = (0, 0, 255) if paused else (0, 255, 0)
    info = f"Vel: {speed}px/f | FPS: {fps:.0f} | Defeitos: {defects_found} | Heatmap: {'ON' if show_heatmap else 'OFF'} | [{status}]"
    cv2.putText(frame, info, (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, status_color, 1)

    # Controles (canto superior direito)
    controls = "[ESPACO] Pausar  [+/-] Vel  [T] Heatmap  [S] Salvar  [Q] Sair"
    text_size = cv2.getTextSize(controls, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)[0]
    cv2.putText(frame, controls, (w - text_size[0] - 10, 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 150, 150), 1)

    return frame


def draw_defect_overlay(frame, binary_mask, confidence_map, show_heatmap,
                        classifier=None, visible_gray=None, device=None):
    """Desenha o overlay de defeitos detectados sobre o frame."""
    h, w = frame.shape[:2]

    if show_heatmap and confidence_map is not None:
        heat_norm = (confidence_map * 255).astype(np.uint8)
        heat_colored = cv2.applyColorMap(heat_norm, cv2.COLORMAP_JET)
        mask_region = (confidence_map > 0.1).astype(np.float32)
        for c in range(3):
            frame[:, :, c] = (frame[:, :, c] * (1 - mask_region * 0.4) +
                              heat_colored[:, :, c] * mask_region * 0.4).astype(np.uint8)

    if binary_mask is not None and binary_mask.sum() > 0:
        defect_pixels = binary_mask > 0
        overlay = frame.copy()
        overlay[defect_pixels] = [0, 0, 255]
        cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)

        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 20:
                x, y, bw, bh = cv2.boundingRect(cnt)
                cv2.rectangle(frame, (x, y), (x + bw, y + bh), (0, 0, 255), 2)

                # Classificar tipo de defeito se disponível
                label = "DEFEITO"
                if classifier is not None and visible_gray is not None:
                    try:
                        pad = 8
                        x1 = max(0, x - pad)
                        y1 = max(0, y - pad)
                        x2 = min(visible_gray.shape[1], x + bw + pad)
                        y2 = min(visible_gray.shape[0], y + bh + pad)
                        patch = visible_gray[y1:y2, x1:x2]
                        if patch.size > 0:
                            patch_resized = cv2.resize(patch, (64, 64))
                            tensor = torch.FloatTensor(patch_resized).unsqueeze(0).unsqueeze(0).to(device)
                            pred = classifier(tensor).argmax(dim=1).item()
                            label = DEFECT_CLASSES[pred]
                    except Exception:
                        label = "DEFEITO"

                cv2.putText(frame, label, (x, y - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

    return frame


def get_current_segment(segments, scroll_x, window_width):
    """Encontra qual segmento do tecido está visível na janela."""
    center_x = scroll_x + window_width // 2
    for seg in segments:
        if seg["x_start"] <= center_x < seg["x_end"]:
            return seg
    return segments[-1] if segments else {"name": "N/A", "defect_type": "N/A", "has_defect": False}


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  SIMULACAO DE ROLO INDUSTRIAL")
    print("  Inspecao Textil em Tempo Real com U-Net")
    print("=" * 60)

    # Carregar modelo
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n  Device: {device}")

    model_path = MODELS_DIR / "unet_best.pth"
    if not model_path.exists():
        print(f"\n  ERRO: Modelo nao encontrado em {model_path}")
        print("  Execute train_unet_gpu_v3.py primeiro!")
        return

    model = UNet(in_channels=1, out_channels=1, features=[32, 64, 128, 256]).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    print(f"  Modelo carregado: {model_path}")

    # Carregar classificador de defeitos (opicional)
    classifier = None
    clf_path = MODELS_DIR / "defect_classifier.pth"
    if clf_path.exists():
        classifier = DefectClassifier(num_classes=len(DEFECT_CLASSES)).to(device)
        classifier.load_state_dict(torch.load(clf_path, map_location=device, weights_only=True))
        classifier.eval()
        print(f"  Classificador carregado: {clf_path}")
    else:
        print(f"  Classificador nao encontrado - mostrando 'DEFEITO' generico")

    # Carregar dataset e montar rolo
    print("\n  Montando rolo de tecido...")
    dataset = AitexDataset(DATASET_DIR)
    roll_img, roll_mask, segments = build_fabric_roll(dataset)
    roll_h, roll_w = roll_img.shape[:2]
    print(f"  Rolo montado: {roll_w} x {roll_h} pixels ({len(segments)} segmentos)")
    print(f"  Comprimento equivalente: {roll_w / 5:.0f} mm")

    # Pré-computar grayscale do rolo inteiro
    roll_gray = cv2.cvtColor(roll_img, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0

    # Estado da simulação
    scroll_x = 0
    speed = SCROLL_SPEED
    paused = False
    show_heatmap = False
    defects_found = 0
    detected_segments = set()
    frame_count = 0
    fps = 0.0
    fps_timer = time.time()

    # Cache de inferência: guardamos o resultado para não recalcular
    last_inference_x = -999
    cached_binary = None
    cached_confidence = None
    INFERENCE_INTERVAL = WINDOW_WIDTH // 4  # Recalcula a cada 1/4 de janela

    cv2.namedWindow("Simulacao Rolo Industrial", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Simulacao Rolo Industrial", WINDOW_WIDTH, WINDOW_HEIGHT + 80)

    print("\n  Simulacao iniciada! Pressione Q ou ESC para sair.\n")

    while True:
        t_frame = time.time()

        # Extrair a janela visível do rolo
        x_start = int(scroll_x) % roll_w
        x_end = x_start + WINDOW_WIDTH

        if x_end <= roll_w:
            visible_img = roll_img[:, x_start:x_end].copy()
            visible_gray = roll_gray[:, x_start:x_end].copy()
            visible_gt = roll_mask[:, x_start:x_end].copy()
        else:
            # Wrap-around: o rolo recomeça
            part1_img = roll_img[:, x_start:]
            part2_img = roll_img[:, :x_end - roll_w]
            visible_img = np.concatenate([part1_img, part2_img], axis=1)

            part1_gray = roll_gray[:, x_start:]
            part2_gray = roll_gray[:, :x_end - roll_w]
            visible_gray = np.concatenate([part1_gray, part2_gray], axis=1)

            part1_gt = roll_mask[:, x_start:]
            part2_gt = roll_mask[:, :x_end - roll_w]
            visible_gt = np.concatenate([part1_gt, part2_gt], axis=1)

        # Inferência U-Net (com cache para performance)
        if abs(scroll_x - last_inference_x) >= INFERENCE_INTERVAL or cached_binary is None:
            cached_binary, cached_confidence = predict_region(
                model, visible_gray, device,
                patch_size=PATCH_SIZE, stride=INFERENCE_STRIDE, threshold=THRESHOLD
            )
            last_inference_x = scroll_x

            # Contar novos defeitos
            current_seg = get_current_segment(segments, int(scroll_x) % roll_w, WINDOW_WIDTH)
            if current_seg["has_defect"] and cached_binary.sum() > 0:
                seg_id = current_seg["name"]
                if seg_id not in detected_segments:
                    detected_segments.add(seg_id)
                    defects_found += 1

        # Construir frame de exibição
        display = visible_img.copy()

        # Redimensionar masks para o tamanho visível se necessário
        bin_display = cached_binary
        conf_display = cached_confidence
        if bin_display.shape[1] != WINDOW_WIDTH:
            bin_display = cv2.resize(bin_display, (WINDOW_WIDTH, WINDOW_HEIGHT), interpolation=cv2.INTER_NEAREST)
            conf_display = cv2.resize(conf_display, (WINDOW_WIDTH, WINDOW_HEIGHT), interpolation=cv2.INTER_LINEAR)

        # Desenhar overlay de defeitos (com classificação se disponível)
        display = draw_defect_overlay(display, bin_display, conf_display, show_heatmap,
                                      classifier=classifier, visible_gray=visible_gray,
                                      device=device)

        # Expandir o frame para caber o HUD
        frame = np.zeros((WINDOW_HEIGHT + 80, WINDOW_WIDTH, 3), dtype=np.uint8)
        frame[40:40 + WINDOW_HEIGHT, :] = display

        # Segmento atual
        current_seg = get_current_segment(segments, int(scroll_x) % roll_w, WINDOW_WIDTH)

        # Ground truth indicator (faixa fina verde/vermelha no topo)
        gt_color = (0, 0, 200) if current_seg["has_defect"] else (0, 180, 0)
        cv2.rectangle(frame, (0, 38), (WINDOW_WIDTH, 40), gt_color, -1)

        # Desenhar HUD
        frame = draw_hud(frame, int(scroll_x) % roll_w, roll_w, speed, paused,
                         show_heatmap, current_seg, defects_found, fps)

        cv2.imshow("Simulacao Rolo Industrial", frame)

        # FPS
        frame_count += 1
        if time.time() - fps_timer >= 1.0:
            fps = frame_count / (time.time() - fps_timer)
            frame_count = 0
            fps_timer = time.time()

        # Avançar scroll
        if not paused:
            scroll_x += speed
            if scroll_x >= roll_w:
                scroll_x = 0  # Loop do rolo

        # Controles
        wait_ms = max(1, int(1000 / FPS_TARGET - (time.time() - t_frame) * 1000))
        key = cv2.waitKey(wait_ms) & 0xFF

        if key == ord('q') or key == 27:  # Q ou ESC
            break
        elif key == ord(' '):  # Espaço = pausar
            paused = not paused
        elif key == ord('+') or key == ord('='):
            speed = min(MAX_SPEED, speed + 1)
        elif key == ord('-') or key == ord('_'):
            speed = max(MIN_SPEED, speed - 1)
        elif key == ord('t') or key == ord('T'):
            show_heatmap = not show_heatmap
        elif key == ord('s') or key == ord('S'):
            save_path = OUTPUT_DIR / f"capture_{int(time.time())}.png"
            cv2.imwrite(str(save_path), frame)
            print(f"  Frame salvo: {save_path}")

    cv2.destroyAllWindows()
    print(f"\n  Simulacao encerrada. Defeitos detectados: {defects_found}")


if __name__ == "__main__":
    main()
