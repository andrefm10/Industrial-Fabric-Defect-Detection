"""
Textile Inspection System — AITEX Dataset Pipeline

Processes the AITEX Fabric Image Database using both classical (PDI)
and deep learning techniques for defect detection and benchmarking.

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
    BENCHMARKS_DIR, DATASET_DIR,
)

from fase1_restauracao.clahe_enhance import CLAHEEnhancer, BrightnessCorrector
from fase1_restauracao.wiener_deconv import WienerRestorer
from fase1_restauracao.fft_filtering import FFTRestorer
from fase1_restauracao.image_slicing import ImageSlicer

from fase2_deteccao.gabor_analysis import GaborAnalyzer
from fase2_deteccao.histogram_otsu import HistogramOtsuDetector
from fase2_deteccao.edge_detection import EdgeDefectDetector
from fase2_deteccao.anomaly_detector import AnomalyDetector

from fase3_benchmark.timer import PipelineProfiler
from fase3_benchmark.metrics import (
    compute_detection_metrics,
    compute_sharpness,
    feasibility_check,
)
from fase3_benchmark.report_generator import ReportGenerator

from utils.visualization import (
    show_comparison,
    show_grid,
    overlay_heatmap,
    plot_histogram_comparison,
    plot_fft_spectrum,
)
from utils.dataset_loader import AitexDataset, DEFECT_TYPE_MAP


# ─── Phase 1: Image Restoration ──────────────────────────────────────────────

def run_phase1(frame: np.ndarray, config: AppConfig, profiler: PipelineProfiler) -> dict:
    """Apply all restoration techniques and return results dict."""
    results = {}

    # Brightness correction
    brightness_corrector = BrightnessCorrector(
        target_mean=config.detection.brightness_target_mean,
        tolerance=config.detection.brightness_tolerance,
    )
    frame = profiler.time_function("brightness_correction", brightness_corrector.auto_correct, frame)
    results["brightness_corrected"] = frame

    # CLAHE
    enhancer = CLAHEEnhancer(
        clip_limit=config.restoration.clahe_clip_limit,
        grid_size=config.restoration.clahe_grid_size,
        sharpening_strength=config.restoration.sharpening_strength,
    )
    results["clahe"] = profiler.time_function("clahe_enhance", enhancer.enhance, frame)

    # Wiener deconvolution
    wiener = WienerRestorer(
        blur_length=config.conveyor.blur_length_px,
        blur_angle=config.restoration.wiener_blur_angle,
        snr=config.restoration.wiener_snr,
    )
    results["wiener"] = profiler.time_function("wiener_deconv", wiener.restore, frame)
    results["wiener_adaptive"] = profiler.time_function("wiener_adaptive", wiener.restore_adaptive, frame)

    # FFT bandpass
    fft = FFTRestorer(
        low_cutoff=config.restoration.fft_cutoff_low,
        high_cutoff=config.restoration.fft_cutoff_high,
    )
    results["fft_bandpass"] = profiler.time_function("fft_bandpass", fft.restore, frame)

    # Image slicing
    slicer = ImageSlicer(
        slice_height=config.restoration.slice_height_px,
        overlap=config.restoration.slice_overlap,
        displacement_px=config.conveyor.speed_mm_per_frame,
    )
    results["slicing"] = profiler.time_function("image_slicing", slicer.restore, frame)

    # Pick best by sharpness
    best_name = max(
        [(k, compute_sharpness(v)) for k, v in results.items() if k != "brightness_corrected"],
        key=lambda x: x[1],
    )[0]
    results["best"] = results[best_name]
    results["best_method"] = best_name

    return results


# ─── Phase 2: Defect Detection ───────────────────────────────────────────────

def run_phase2(
    image: np.ndarray,
    config: AppConfig,
    profiler: PipelineProfiler,
    ground_truth_mask: np.ndarray = None,
    anomaly_detector: AnomalyDetector = None,
) -> dict:
    """Run all detection methods and return results + metrics."""
    results = {}
    metrics = {}

    # Gabor filter bank
    gabor = GaborAnalyzer(
        orientations=config.detection.gabor_orientations,
        frequencies=config.detection.gabor_frequencies,
        sigma=config.detection.gabor_sigma,
    )
    results["gabor_mask"] = profiler.time_function("gabor_detection", gabor.detect_anomalies, image)

    # Histogram + Otsu
    otsu = HistogramOtsuDetector(
        blur_kernel=config.detection.otsu_blur_kernel,
        min_area=config.detection.min_defect_area_px,
    )
    results["otsu_mask"], results["otsu_contours"] = profiler.time_function(
        "otsu_detection", otsu.detect_defects, image
    )

    # Edge detection + classification
    edge_det = EdgeDefectDetector(
        canny_sigma=config.detection.canny_sigma,
        morphology_kernel=config.detection.morphology_kernel_size,
        min_area=config.detection.min_defect_area_px,
    )
    results["edge_mask"], results["edge_defects"] = profiler.time_function(
        "edge_detection", edge_det.detect_and_classify, image
    )
    results["edge_annotated"] = edge_det.annotate_image(image, results["edge_defects"])

    # Anomaly detector (autoencoder or statistical fallback)
    if anomaly_detector is None:
        anomaly_detector = AnomalyDetector(patch_size=64, threshold_percentile=95.0)
    results["anomaly_mask"], results["anomaly_heatmap"] = profiler.time_function(
        "anomaly_detection", anomaly_detector.detect, image
    )

    # Compute metrics against ground truth if available
    if ground_truth_mask is not None:
        for method_key in ["gabor_mask", "otsu_mask", "edge_mask", "anomaly_mask"]:
            gt_resized = ground_truth_mask
            pred = results[method_key]
            # Ensure same size
            if pred.shape[:2] != gt_resized.shape[:2]:
                gt_resized = cv2.resize(gt_resized, (pred.shape[1], pred.shape[0]),
                                         interpolation=cv2.INTER_NEAREST)
            m = compute_detection_metrics(pred, gt_resized)
            metrics[method_key] = m

    results["metrics"] = metrics
    return results


# ─── Dataset Batch Processing ────────────────────────────────────────────────

def run_dataset_pipeline(config: AppConfig, no_viz: bool = False):
    """
    Process the entire AITEX dataset in batch:
    1. Load dataset index
    2. Train autoencoder on normal images
    3. For each defect image: restore → detect → measure
    4. Aggregate metrics and generate report
    """
    print("=" * 60)
    print("  TEXTILE INSPECTION SYSTEM")
    print("  Mode: AITEX Dataset Batch Processing")
    print("=" * 60)

    # ── Load dataset ──
    dataset = AitexDataset(DATASET_DIR)
    print(f"\n{dataset.summary()}\n")

    # ── Train anomaly detector on normal images ──
    print("=" * 60)
    print("TRAINING: Anomaly Detector on normal fabric samples")
    print("=" * 60)

    anomaly_detector = AnomalyDetector(patch_size=64, threshold_percentile=95.0)
    normal_images = dataset.load_normal_images(max_count=10)
    if normal_images:
        print(f"  Training on {len(normal_images)} normal images...")
        anomaly_detector.train(normal_images, epochs=15, batch_size=32)
        print("  Training complete.\n")
    else:
        print("  No normal images found. Using statistical fallback.\n")

    # ── Batch processing ──
    print("=" * 60)
    print("BATCH PROCESSING: All defect images")
    print("=" * 60)

    profiler = PipelineProfiler()

    # Aggregated metrics per method per defect type
    # Structure: {defect_type: {method: [metric_dicts]}}
    metrics_by_type = defaultdict(lambda: defaultdict(list))
    # Overall metrics per method
    metrics_overall = defaultdict(list)
    # Timing data
    all_timings = []
    # Per-image results for the report (store a few examples)
    example_results = []
    restoration_methods_sharpness = defaultdict(list)

    total = dataset.num_defect
    samples_with_masks = 0

    for i, (sample, image, mask) in enumerate(dataset.iter_defect_with_masks()):
        progress = f"[{i+1}/{total}]"
        print(f"  {progress} {sample.image_path.name} — {sample.defect_name}", end="")

        img_profiler = PipelineProfiler()

        # Phase 1: Restoration
        restoration = run_phase1(image, config, img_profiler)
        best_method = restoration["best_method"]
        restored = restoration["best"]

        # Track sharpness for each restoration method
        for method_name in ["clahe", "wiener", "wiener_adaptive", "fft_bandpass", "slicing"]:
            if method_name in restoration:
                sharpness = compute_sharpness(restoration[method_name])
                restoration_methods_sharpness[method_name].append(sharpness)

        # Phase 2: Detection
        detection = run_phase2(
            restored, config, img_profiler,
            ground_truth_mask=mask,
            anomaly_detector=anomaly_detector,
        )

        # Collect metrics
        samples_with_masks += 1
        for method_key, method_metrics in detection["metrics"].items():
            method_name = method_key.replace("_mask", "")
            metrics_by_type[sample.defect_name][method_name].append(method_metrics)
            metrics_overall[method_name].append(method_metrics)

        # Timing
        timing = img_profiler.summary()
        total_ms = sum(r.avg_ms for r in timing)
        all_timings.append(total_ms)

        print(f" — best: {best_method}, time: {total_ms:.0f}ms")

        # Save a few examples for visualization (one per defect type)
        seen_types = {e["defect_type"] for e in example_results}
        if sample.defect_name not in seen_types and len(example_results) < 8:
            example_results.append({
                "defect_type": sample.defect_name,
                "filename": sample.image_path.name,
                "original": image,
                "restored": restored,
                "best_method": best_method,
                "mask_gt": mask,
                "detection": detection,
            })

    # ── Aggregate & Report ──
    print(f"\n{'=' * 60}")
    print("RESULTS SUMMARY")
    print(f"{'=' * 60}")
    print(f"  Images processed:  {samples_with_masks}")
    print(f"  Avg time/image:    {np.mean(all_timings):.0f} ms")
    print(f"  Avg FPS:           {1000.0 / np.mean(all_timings):.1f}")

    print(f"\n  Overall Detection Metrics (avg across all images):")
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

    # Restoration summary
    restoration_summary = {}
    for method, sharpness_list in restoration_methods_sharpness.items():
        restoration_summary[method] = {
            "avg_sharpness": np.mean(sharpness_list),
            "std_sharpness": np.std(sharpness_list),
        }

    # ── Generate Report ──
    print(f"\n{'=' * 60}")
    print("GENERATING REPORT")
    print(f"{'=' * 60}")

    report = ReportGenerator(BENCHMARKS_DIR)

    report.add_dataset_overview(dataset)
    report.add_restoration_summary(restoration_summary)
    report.add_overall_detection_chart(overall_summary)
    report.add_per_defect_heatmap(type_summary)
    report.add_example_gallery(example_results)
    report.add_timing_summary(all_timings)

    avg_fps = 1000.0 / np.mean(all_timings) if all_timings else 0
    feas = feasibility_check(fps=avg_fps, target_fps=config.benchmark.target_fps,
                              conveyor_speed_m_min=config.conveyor.speed_m_per_min)
    report.add_feasibility_summary(feas)

    # Raw data JSON
    raw_data = {
        "overall_metrics": {k: {kk: round(vv, 4) for kk, vv in v.items()} for k, v in overall_summary.items()},
        "per_defect_type": {
            dt: {m: {kk: round(vv, 4) for kk, vv in mv.items()} for m, mv in methods.items()}
            for dt, methods in type_summary.items()
        },
        "restoration": {k: {kk: round(vv, 2) for kk, vv in v.items()} for k, v in restoration_summary.items()},
        "timing": {
            "avg_ms": round(np.mean(all_timings), 1),
            "std_ms": round(np.std(all_timings), 1),
            "min_ms": round(np.min(all_timings), 1),
            "max_ms": round(np.max(all_timings), 1),
        },
        "feasibility": {k: round(v, 2) if isinstance(v, float) else v for k, v in feas.items()},
    }

    report.generate_html(
        title="AITEX Dataset — Textile Defect Detection Benchmark",
        extra_data=raw_data,
    )

    print(f"\n✅ Pipeline complete. Results saved to: {RESULTS_DIR}")


# ─── Single Image Mode ───────────────────────────────────────────────────────

def run_single_image(image_path: str, config: AppConfig, no_viz: bool = False):
    """Process a single image through the full pipeline."""
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
    print(f"  Mode: Single Image Analysis")
    print("=" * 60)
    print(f"Input: {path}")
    print(f"Image size: {image.shape[1]}x{image.shape[0]}")

    # Check if there's a corresponding mask
    mask = None
    mask_path = DATASET_DIR / "Mask_images" / f"{path.stem}_mask.png"
    if mask_path.exists():
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        print(f"Ground truth mask: {mask_path.name}")

    profiler = PipelineProfiler()

    # Phase 1
    print(f"\n{'=' * 60}")
    print("PHASE 1: IMAGE RESTORATION")
    print(f"{'=' * 60}")
    restoration = run_phase1(image, config, profiler)
    for method in ["clahe", "wiener", "wiener_adaptive", "fft_bandpass", "slicing"]:
        if method in restoration:
            print(f"  [✓] {method} — sharpness: {compute_sharpness(restoration[method]):.1f}")
    print(f"\n  ★ Best restoration: {restoration['best_method']}")

    # Phase 2
    print(f"\n{'=' * 60}")
    print("PHASE 2: DEFECT DETECTION")
    print(f"{'=' * 60}")
    detection = run_phase2(restoration["best"], config, profiler, ground_truth_mask=mask)

    for method in ["gabor_mask", "otsu_mask", "edge_mask", "anomaly_mask"]:
        px = (detection[method] > 0).sum()
        print(f"  [✓] {method.replace('_mask','')} — anomaly area: {px} px")
        if method in detection["metrics"]:
            m = detection["metrics"][method]
            print(f"       IoU: {m['iou']:.3f} | F1: {m['f1_score']:.3f}")

    if "edge_defects" in detection:
        for d in detection["edge_defects"]:
            print(f"      → {d['label']} (area: {d['area']:.0f} px)")

    # Timing
    timing = profiler.summary()
    total_ms = sum(r.avg_ms for r in timing)
    print(f"\n  Total time: {total_ms:.0f} ms ({1000/total_ms:.1f} FPS)")

    # Visualization
    if not no_viz:
        titles = ["Input"]
        images = [image]
        for name in ["clahe", "wiener", "fft_bandpass", "slicing"]:
            if name in restoration:
                images.append(restoration[name])
                titles.append(name.replace("_", " ").title())
        show_grid(images, titles, cols=3, save_path=RESTORED_DIR / "phase1_comparison.png")

        det_images = [restoration["best"]]
        det_titles = ["Restored (Best)"]
        if "edge_annotated" in detection:
            det_images.append(detection["edge_annotated"])
            det_titles.append("Edge Detection")
        if "anomaly_heatmap" in detection:
            heatmap = overlay_heatmap(restoration["best"],
                                       detection["anomaly_heatmap"].astype(np.float32))
            det_images.append(heatmap)
            det_titles.append("Anomaly Heatmap")
        show_grid(det_images, det_titles, cols=2, save_path=DEFECTS_DIR / "phase2_comparison.png")

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
    gt_mask = sample["masks"].get("combined", np.zeros(image.shape[:2], dtype=np.uint8))

    profiler = PipelineProfiler()
    restoration = run_phase1(image, config, profiler)
    detection = run_phase2(restoration["best"], config, profiler, ground_truth_mask=gt_mask)

    timing = profiler.summary()
    total_ms = sum(r.avg_ms for r in timing)
    print(f"\nTotal time: {total_ms:.0f} ms ({1000/total_ms:.1f} FPS)")
    print(f"✅ Demo complete. Results saved to: {RESULTS_DIR}")


# ─── Entry Point ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Textile Inspection System — AITEX Dataset Pipeline"
    )
    parser.add_argument("--dataset", action="store_true",
                        help="Process entire AITEX dataset in batch mode")
    parser.add_argument("--image", type=str,
                        help="Process a single image file")
    parser.add_argument("--demo", action="store_true",
                        help="Run with synthetic fabric sample")
    parser.add_argument("--no-viz", action="store_true",
                        help="Skip visualization (save only)")
    args = parser.parse_args()

    config = AppConfig()

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