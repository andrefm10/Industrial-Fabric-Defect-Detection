"""
Textile Inspection System — AITEX Dataset Pipeline

Pipeline focado em Deep Learning (U-Net) para detecção de defeitos
têxteis, com pré-processamento clássico leve (CLAHE).

Usage:
    python3 main.py --dataset           # Process entire AITEX dataset
    python3 main.py --image <path>      # Process a single image
    python3 main.py --demo              # Run with synthetic sample
"""

import argparse
import sys
import json
from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np

from config import (
    AppConfig, RESULTS_DIR, RESTORED_DIR, DEFECTS_DIR,
    BENCHMARKS_DIR, DATASET_DIR, MODELS_DIR,
)

# ── Fase 1: Pré-processamento leve ──
from fase1_restauracao.clahe_enhance import CLAHEEnhancer, BrightnessCorrector

# ── Fase 2: Detecção por Deep Learning ──
from fase2_deteccao.unet_segmentation import UNetSegmenter

# ── Fase 3: Benchmark ──
from fase3_benchmark.timer import PipelineProfiler
from fase3_benchmark.metrics import (
    compute_detection_metrics,
    compute_sharpness,
    feasibility_check,
)
from fase3_benchmark.report_generator import ReportGenerator

# ── Utilitários ──
from utils.visualization import show_grid, overlay_heatmap
from utils.dataset_loader import AitexDataset, DEFECT_TYPE_MAP


# ─── Visão Clássica (desativada — manter para comparação futura) ─────────────
# from fase1_restauracao.wiener_deconv import WienerRestorer
# from fase1_restauracao.fft_filtering import FFTRestorer
# from fase1_restauracao.image_slicing import ImageSlicer
# from fase2_deteccao.gabor_analysis import GaborAnalyzer
# from fase2_deteccao.histogram_otsu import HistogramOtsuDetector
# from fase2_deteccao.edge_detection import EdgeDefectDetector
# from fase2_deteccao.anomaly_detector import AnomalyDetector


# ─── Phase 1: Pré-processamento ──────────────────────────────────────────────

def run_preprocessing(frame: np.ndarray, config: AppConfig,
                      profiler: PipelineProfiler) -> np.ndarray:
    """Pré-processamento leve: correção de brilho + CLAHE."""
    # Correção de brilho
    brightness = BrightnessCorrector(
        target_mean=config.detection.brightness_target_mean,
        tolerance=config.detection.brightness_tolerance,
    )
    frame = profiler.time_function("brightness_correction",
                                   brightness.auto_correct, frame)

    # CLAHE para normalização de contraste
    enhancer = CLAHEEnhancer(
        clip_limit=config.restoration.clahe_clip_limit,
        grid_size=config.restoration.clahe_grid_size,
        sharpening_strength=config.restoration.sharpening_strength,
    )
    result = profiler.time_function("clahe_enhance", enhancer.enhance, frame)
    return result


# ─── Phase 2: Detecção por U-Net ─────────────────────────────────────────────

def run_detection_unet(
    image: np.ndarray,
    config: AppConfig,
    profiler: PipelineProfiler,
    unet: UNetSegmenter,
    ground_truth_mask: np.ndarray = None,
) -> dict:
    """Detecção de defeitos usando U-Net."""
    results = {}
    metrics = {}

    # U-Net prediction
    unet_mask, confidence = profiler.time_function(
        "unet_segmentation", unet.predict, image,
        config.deep_learning.unet_threshold,
    )
    results["unet_mask"] = unet_mask
    results["unet_confidence"] = confidence

    # Métricas contra ground truth
    if ground_truth_mask is not None:
        gt = ground_truth_mask
        pred = unet_mask
        if pred.shape[:2] != gt.shape[:2]:
            gt = cv2.resize(gt, (pred.shape[1], pred.shape[0]),
                            interpolation=cv2.INTER_NEAREST)
        metrics["unet"] = compute_detection_metrics(pred, gt)

    results["metrics"] = metrics
    return results


# ─── Dataset Batch Processing ────────────────────────────────────────────────

def run_dataset_pipeline(config: AppConfig, no_viz: bool = False):
    """
    Pipeline principal com U-Net:
    1. Carrega dataset AITEX
    2. Treina U-Net nas imagens (com split treino/validação)
    3. Roda inferência em todas as imagens com defeito
    4. Agrega métricas e gera relatório
    """
    dl = config.deep_learning

    print("=" * 60)
    print("  TEXTILE INSPECTION SYSTEM")
    print("  Mode: AITEX Dataset — Deep Learning Pipeline (U-Net)")
    print("=" * 60)

    # ── Carregar dataset ──
    dataset = AitexDataset(DATASET_DIR)
    print(f"\n{dataset.summary()}\n")

    # ── Preparar dados para treino da U-Net ──
    print("=" * 60)
    print("PREPARING: Loading images for U-Net training")
    print("=" * 60)

    defect_images = []
    defect_masks = []
    for sample, image, mask in dataset.iter_defect_with_masks():
        defect_images.append(image)
        defect_masks.append(mask)

    normal_images = dataset.load_normal_images(
        max_count=dl.max_normal_images)

    print(f"  Defect images:  {len(defect_images)}")
    print(f"  Normal images:  {len(normal_images)}")

    # ── Treinar U-Net ──
    print(f"\n{'=' * 60}")
    print("TRAINING: U-Net Semantic Segmentation")
    print("=" * 60)

    model_path = MODELS_DIR / "unet_best.pth"
    unet = UNetSegmenter(
        patch_size=dl.unet_patch_size,
        model_path=model_path if model_path.exists() else None,
    )

    if not unet.is_trained:
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
    else:
        print("  U-Net already trained — using saved model.")

    # ── Inferência em todas as imagens com defeito ──
    print(f"\n{'=' * 60}")
    print("INFERENCE: Running U-Net on all defect images")
    print("=" * 60)

    profiler = PipelineProfiler()
    metrics_overall = defaultdict(list)
    metrics_by_type = defaultdict(lambda: defaultdict(list))
    all_timings = []
    example_results = []

    total = len(defect_images)
    for i, (sample, image, mask) in enumerate(dataset.iter_defect_with_masks()):
        progress = f"[{i+1}/{total}]"

        img_profiler = PipelineProfiler()

        # Phase 1: pré-processamento
        preprocessed = run_preprocessing(image, config, img_profiler)

        # Phase 2: U-Net
        detection = run_detection_unet(
            preprocessed, config, img_profiler,
            unet=unet, ground_truth_mask=mask,
        )

        # Coletar métricas
        for method_key, m in detection["metrics"].items():
            metrics_by_type[sample.defect_name][method_key].append(m)
            metrics_overall[method_key].append(m)

        # Timing
        timing = img_profiler.summary()
        total_ms = sum(r.avg_ms for r in timing)
        all_timings.append(total_ms)

        # Print progress
        m = detection["metrics"].get("unet", {})
        f1 = m.get("f1_score", 0)
        iou = m.get("iou", 0)
        print(f"  {progress} {sample.image_path.name} — "
              f"{sample.defect_name} — "
              f"F1: {f1:.3f} | IoU: {iou:.3f} | {total_ms:.0f}ms")

        # Guardar exemplos para galeria
        seen_types = {e["defect_type"] for e in example_results}
        if sample.defect_name not in seen_types and len(example_results) < 8:
            example_results.append({
                "defect_type": sample.defect_name,
                "filename": sample.image_path.name,
                "original": image,
                "restored": preprocessed,
                "best_method": "clahe",
                "mask_gt": mask,
                "detection": detection,
            })

    # ── Resultados Agregados ──
    print(f"\n{'=' * 60}")
    print("RESULTS SUMMARY")
    print(f"{'=' * 60}")
    print(f"  Images processed:  {len(all_timings)}")
    if all_timings:
        print(f"  Avg time/image:    {np.mean(all_timings):.0f} ms")
        print(f"  Avg FPS:           {1000.0 / np.mean(all_timings):.1f}")

    print(f"\n  U-Net Detection Metrics (avg across all images):")
    overall_summary = {}
    for method, metric_list in metrics_overall.items():
        avg_iou = np.mean([m["iou"] for m in metric_list])
        avg_f1 = np.mean([m["f1_score"] for m in metric_list])
        avg_prec = np.mean([m["precision"] for m in metric_list])
        avg_rec = np.mean([m["recall"] for m in metric_list])
        overall_summary[method] = {
            "iou": avg_iou, "f1": avg_f1,
            "precision": avg_prec, "recall": avg_rec,
        }
        print(f"    {method:12s} — F1: {avg_f1:.3f} | IoU: {avg_iou:.3f} | "
              f"Prec: {avg_prec:.3f} | Rec: {avg_rec:.3f}")

    # Per-defect-type breakdown
    type_summary = {}
    for defect_type, methods in metrics_by_type.items():
        type_summary[defect_type] = {}
        for method, metric_list in methods.items():
            type_summary[defect_type][method] = {
                "iou": np.mean([m["iou"] for m in metric_list]),
                "f1": np.mean([m["f1_score"] for m in metric_list]),
                "precision": np.mean([m["precision"] for m in metric_list]),
                "recall": np.mean([m["recall"] for m in metric_list]),
                "count": len(metric_list),
            }

    # ── Gerar Relatório ──
    print(f"\n{'=' * 60}")
    print("GENERATING REPORT")
    print(f"{'=' * 60}")

    report = ReportGenerator(BENCHMARKS_DIR)
    report.add_dataset_overview(dataset)
    report.add_overall_detection_chart(overall_summary)
    report.add_per_defect_heatmap(type_summary)
    report.add_example_gallery(example_results)

    if all_timings:
        report.add_timing_summary(all_timings)
        avg_fps = 1000.0 / np.mean(all_timings)
        feas = feasibility_check(
            fps=avg_fps,
            target_fps=config.benchmark.target_fps,
            conveyor_speed_m_min=config.conveyor.speed_m_per_min,
        )
        report.add_feasibility_summary(feas)
    else:
        feas = {}

    # Raw data JSON
    raw_data = {
        "pipeline": "Deep Learning (U-Net)",
        "overall_metrics": {
            k: {kk: round(vv, 4) for kk, vv in v.items()}
            for k, v in overall_summary.items()
        },
        "per_defect_type": {
            dt: {m: {kk: round(vv, 4) for kk, vv in mv.items()}
                 for m, mv in methods.items()}
            for dt, methods in type_summary.items()
        },
        "timing": {
            "avg_ms": round(np.mean(all_timings), 1) if all_timings else 0,
            "std_ms": round(np.std(all_timings), 1) if all_timings else 0,
            "min_ms": round(np.min(all_timings), 1) if all_timings else 0,
            "max_ms": round(np.max(all_timings), 1) if all_timings else 0,
        },
        "feasibility": {
            k: round(v, 2) if isinstance(v, float) else v
            for k, v in feas.items()
        },
    }

    report.generate_html(
        title="AITEX Dataset — U-Net Defect Segmentation Benchmark",
        extra_data=raw_data,
    )

    print(f"\n✅ Pipeline complete. Results saved to: {RESULTS_DIR}")


# ─── Single Image Mode ───────────────────────────────────────────────────────

def run_single_image(image_path: str, config: AppConfig,
                     no_viz: bool = False):
    """Process a single image with U-Net."""
    path = Path(image_path)
    if not path.exists():
        print(f"Image not found: {path}")
        sys.exit(1)

    image = cv2.imread(str(path))
    if image is None:
        print(f"Failed to load image: {path}")
        sys.exit(1)

    print("=" * 60)
    print("  TEXTILE INSPECTION SYSTEM")
    print("  Mode: Single Image Analysis (U-Net)")
    print("=" * 60)
    print(f"Input: {path}")
    print(f"Image size: {image.shape[1]}x{image.shape[0]}")

    # Load trained model
    model_path = MODELS_DIR / "unet_best.pth"
    unet = UNetSegmenter(
        patch_size=config.deep_learning.unet_patch_size,
        model_path=model_path,
    )

    if not unet.is_trained:
        print("\n⚠ No trained U-Net model found.")
        print("  Run: python3 main.py --dataset  (to train first)")
        sys.exit(1)

    # Check for ground truth mask
    mask = None
    mask_path = DATASET_DIR / "Mask_images" / f"{path.stem}_mask.png"
    if mask_path.exists():
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        print(f"Ground truth mask: {mask_path.name}")

    profiler = PipelineProfiler()

    # Phase 1: pré-processamento
    print(f"\n{'=' * 60}")
    print("PHASE 1: PRE-PROCESSING")
    print(f"{'=' * 60}")
    preprocessed = run_preprocessing(image, config, profiler)
    print(f"  [✓] CLAHE — sharpness: {compute_sharpness(preprocessed):.1f}")

    # Phase 2: U-Net
    print(f"\n{'=' * 60}")
    print("PHASE 2: U-NET SEGMENTATION")
    print(f"{'=' * 60}")
    detection = run_detection_unet(
        preprocessed, config, profiler,
        unet=unet, ground_truth_mask=mask,
    )

    unet_mask = detection["unet_mask"]
    px = (unet_mask > 0).sum()
    print(f"  [✓] U-Net — anomaly area: {px} px")
    if "unet" in detection["metrics"]:
        m = detection["metrics"]["unet"]
        print(f"       IoU: {m['iou']:.3f} | F1: {m['f1_score']:.3f} | "
              f"Prec: {m['precision']:.3f} | Rec: {m['recall']:.3f}")

    # Timing
    timing = profiler.summary()
    total_ms = sum(r.avg_ms for r in timing)
    print(f"\n  Total time: {total_ms:.0f} ms ({1000/total_ms:.1f} FPS)")

    # Visualization
    if not no_viz:
        heatmap = overlay_heatmap(
            preprocessed, detection["unet_confidence"])
        show_grid(
            [image, preprocessed, heatmap],
            ["Original", "Pre-processed (CLAHE)", "U-Net Defect Heatmap"],
            cols=3,
            save_path=DEFECTS_DIR / "unet_result.png",
        )

    print(f"\n✅ Done. Results saved to: {RESULTS_DIR}")


# ─── Demo Mode ────────────────────────────────────────────────────────────────

def run_demo(config: AppConfig, no_viz: bool = False):
    """Run with synthetic fabric sample."""
    from utils.sample_generator import generate_test_sample

    print("=" * 60)
    print("  TEXTILE INSPECTION SYSTEM")
    print("  Mode: DEMO (Synthetic Fabric)")
    print("=" * 60)

    sample = generate_test_sample(
        size=(512, 512),
        blur_length=config.conveyor.blur_length_px,
        defects=True,
    )
    image = sample["blurred"]
    gt_mask = sample["masks"].get(
        "combined", np.zeros(image.shape[:2], dtype=np.uint8))

    # Check for trained model
    model_path = MODELS_DIR / "unet_best.pth"
    unet = UNetSegmenter(
        patch_size=config.deep_learning.unet_patch_size,
        model_path=model_path,
    )

    profiler = PipelineProfiler()
    preprocessed = run_preprocessing(image, config, profiler)

    if unet.is_trained:
        detection = run_detection_unet(
            preprocessed, config, profiler,
            unet=unet, ground_truth_mask=gt_mask,
        )
        m = detection["metrics"].get("unet", {})
        print(f"\n  U-Net F1: {m.get('f1_score', 0):.3f} | "
              f"IoU: {m.get('iou', 0):.3f}")
    else:
        print("\n  ⚠ No trained U-Net model. Run --dataset first.")

    timing = profiler.summary()
    total_ms = sum(r.avg_ms for r in timing)
    print(f"\nTotal time: {total_ms:.0f} ms ({1000/total_ms:.1f} FPS)")
    print(f"✅ Demo complete. Results saved to: {RESULTS_DIR}")


# ─── Entry Point ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Textile Inspection System — U-Net Pipeline"
    )
    parser.add_argument("--dataset", action="store_true",
                        help="Process entire AITEX dataset in batch mode")
    parser.add_argument("--image", type=str,
                        help="Process a single image file")
    parser.add_argument("--demo", action="store_true",
                        help="Run with synthetic fabric sample")
    parser.add_argument("--no-viz", action="store_true",
                        help="Skip visualization (save only)")
    parser.add_argument("--retrain", action="store_true",
                        help="Force retrain even if model exists")
    args = parser.parse_args()

    config = AppConfig()

    # Delete saved model if --retrain
    if args.retrain:
        model_path = MODELS_DIR / "unet_best.pth"
        if model_path.exists():
            model_path.unlink()
            print("  Deleted saved model. Will retrain.")

    if args.dataset:
        run_dataset_pipeline(config, no_viz=args.no_viz)
    elif args.image:
        run_single_image(args.image, config, no_viz=args.no_viz)
    elif args.demo:
        run_demo(config, no_viz=args.no_viz)
    else:
        print("No input specified. Use --dataset, --image <path>, or --demo.")
        print("\nRecommended: python3 main.py --dataset")
        sys.exit(1)


if __name__ == "__main__":
    main()