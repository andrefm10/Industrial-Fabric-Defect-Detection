"""
Report Generator — AITEX Dataset Benchmark

Generates a comprehensive HTML report with charts:
  - Dataset overview (defect distribution)
  - Restoration comparison (sharpness by method)
  - Detection metrics (F1/IoU by method)
  - Per-defect-type heatmap
  - Example gallery with ground truth comparison
  - Timing analysis
  - Feasibility check
"""

import json
import datetime
from pathlib import Path
from typing import List, Optional, Dict

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from fase3_benchmark.timer import TimingResult


# ── Color palette ──
COLORS = {
    "primary": "#3498db",
    "secondary": "#2ecc71",
    "accent": "#e74c3c",
    "warning": "#f39c12",
    "purple": "#9b59b6",
    "teal": "#1abc9c",
    "dark": "#2c3e50",
    "gray": "#95a5a6",
}

METHOD_COLORS = {
    "gabor": "#3498db",
    "otsu": "#2ecc71",
    "edge": "#e74c3c",
    "anomaly": "#9b59b6",
}

RESTORATION_COLORS = {
    "clahe": "#3498db",
    "wiener": "#2ecc71",
    "wiener_adaptive": "#1abc9c",
    "fft_bandpass": "#e74c3c",
    "slicing": "#f39c12",
}


class ReportGenerator:
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.sections: list[str] = []
        self.images_saved: list[Path] = []

    def _save_figure(self, fig, name: str) -> str:
        path = self.output_dir / f"{name}.png"
        fig.savefig(str(path), dpi=150, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        self.images_saved.append(path)
        return f"{name}.png"

    # ── Dataset Overview ──

    def add_dataset_overview(self, dataset) -> str:
        """Bar chart showing defect type distribution."""
        defect_types = dataset.defect_types
        names = list(defect_types.keys())
        counts = list(defect_types.values())

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Defect distribution
        colors = plt.cm.Set2(np.linspace(0, 1, len(names)))
        bars = ax1.barh(names[::-1], counts[::-1], color=colors[::-1], edgecolor="white", linewidth=0.5)
        ax1.set_xlabel("Number of Images", fontsize=11)
        ax1.set_title("Defect Type Distribution", fontsize=13, fontweight="bold")
        for bar, val in zip(bars, counts[::-1]):
            ax1.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                     str(val), va="center", fontsize=10, fontweight="bold")
        ax1.spines["top"].set_visible(False)
        ax1.spines["right"].set_visible(False)

        # Pie chart: defect vs normal
        sizes = [dataset.num_defect, dataset.num_normal]
        labels = [f"With Defect\n({dataset.num_defect})", f"Normal\n({dataset.num_normal})"]
        pie_colors = [COLORS["accent"], COLORS["secondary"]]
        ax2.pie(sizes, labels=labels, colors=pie_colors, autopct="%1.0f%%",
                startangle=90, textprops={"fontsize": 11},
                wedgeprops={"edgecolor": "white", "linewidth": 2})
        ax2.set_title("Dataset Composition", fontsize=13, fontweight="bold")

        fig.suptitle("AITEX Fabric Image Database", fontsize=15, fontweight="bold", y=1.02)
        plt.tight_layout()
        return self._save_figure(fig, "dataset_overview")

    # ── Restoration Summary ──

    def add_restoration_summary(self, restoration_summary: dict) -> str:
        """Bar chart comparing average sharpness across restoration methods."""
        methods = list(restoration_summary.keys())
        avg_sharpness = [restoration_summary[m]["avg_sharpness"] for m in methods]
        std_sharpness = [restoration_summary[m]["std_sharpness"] for m in methods]

        fig, ax = plt.subplots(figsize=(10, 5))

        colors = [RESTORATION_COLORS.get(m, COLORS["gray"]) for m in methods]
        display_names = [m.replace("_", " ").title() for m in methods]

        bars = ax.bar(display_names, avg_sharpness, yerr=std_sharpness,
                      color=colors, edgecolor="white", linewidth=0.5,
                      capsize=5, error_kw={"linewidth": 1.5})

        best_idx = np.argmax(avg_sharpness)
        bars[best_idx].set_edgecolor("#2c3e50")
        bars[best_idx].set_linewidth(2.5)

        for bar, val in zip(bars, avg_sharpness):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(std_sharpness) * 0.3,
                    f"{val:.0f}", ha="center", fontsize=10, fontweight="bold")

        ax.set_ylabel("Average Sharpness (Laplacian Variance)", fontsize=11)
        ax.set_title("Phase 1 — Restoration Method Comparison", fontsize=13, fontweight="bold")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # Add legend for best method
        ax.annotate(f"★ Best: {display_names[best_idx]}",
                    xy=(best_idx, avg_sharpness[best_idx]),
                    xytext=(best_idx + 0.5, avg_sharpness[best_idx] * 1.15),
                    fontsize=10, fontweight="bold", color=COLORS["dark"],
                    arrowprops=dict(arrowstyle="->", color=COLORS["dark"]))

        plt.tight_layout()
        return self._save_figure(fig, "restoration_summary")

    # ── Overall Detection Chart ──

    def add_overall_detection_chart(self, overall_summary: dict) -> str:
        """Grouped bar chart: F1, IoU, Precision, Recall per detection method."""
        methods = list(overall_summary.keys())
        display_names = [m.replace("_", " ").title() for m in methods]

        metric_keys = ["f1", "iou", "precision", "recall"]
        metric_labels = ["F1 Score", "IoU", "Precision", "Recall"]

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # LEFT: Grouped bar chart for F1 and IoU
        x = np.arange(len(methods))
        width = 0.35

        f1_values = [overall_summary[m]["f1"] for m in methods]
        iou_values = [overall_summary[m]["iou"] for m in methods]

        bars1 = axes[0].bar(x - width/2, f1_values, width, label="F1 Score",
                            color=COLORS["primary"], edgecolor="white")
        bars2 = axes[0].bar(x + width/2, iou_values, width, label="IoU",
                            color=COLORS["secondary"], edgecolor="white")

        axes[0].set_ylabel("Score", fontsize=11)
        axes[0].set_title("Detection Performance", fontsize=13, fontweight="bold")
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(display_names, fontsize=10)
        axes[0].legend(fontsize=10)
        axes[0].set_ylim(0, 1.0)
        axes[0].spines["top"].set_visible(False)
        axes[0].spines["right"].set_visible(False)

        # Add value labels
        for bar, val in zip(bars1, f1_values):
            axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                         f"{val:.3f}", ha="center", fontsize=8, fontweight="bold")
        for bar, val in zip(bars2, iou_values):
            axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                         f"{val:.3f}", ha="center", fontsize=8, fontweight="bold")

        # RIGHT: Precision vs Recall scatter
        prec_values = [overall_summary[m]["precision"] for m in methods]
        rec_values = [overall_summary[m]["recall"] for m in methods]
        colors = [METHOD_COLORS.get(m, COLORS["gray"]) for m in methods]

        for i, m in enumerate(methods):
            axes[1].scatter(rec_values[i], prec_values[i], s=200, c=colors[i],
                           edgecolors="white", linewidth=2, zorder=5, label=display_names[i])
            axes[1].annotate(display_names[i],
                            (rec_values[i], prec_values[i]),
                            textcoords="offset points", xytext=(10, 5),
                            fontsize=9, fontweight="bold")

        axes[1].set_xlabel("Recall", fontsize=11)
        axes[1].set_ylabel("Precision", fontsize=11)
        axes[1].set_title("Precision vs Recall", fontsize=13, fontweight="bold")
        axes[1].set_xlim(-0.05, 1.05)
        axes[1].set_ylim(-0.05, 1.05)
        axes[1].plot([0, 1], [0, 1], "k--", alpha=0.2, linewidth=1)
        axes[1].spines["top"].set_visible(False)
        axes[1].spines["right"].set_visible(False)
        axes[1].grid(True, alpha=0.3)

        fig.suptitle("Phase 2 — Detection Methods Comparison", fontsize=15, fontweight="bold", y=1.02)
        plt.tight_layout()
        return self._save_figure(fig, "detection_overall")

    # ── Per-Defect-Type Heatmap ──

    def add_per_defect_heatmap(self, type_summary: dict) -> str:
        """Heatmap: F1 score per method (columns) × defect type (rows)."""
        defect_types = sorted(type_summary.keys())
        methods = sorted(set(
            m for dt in type_summary.values() for m in dt.keys()
        ))
        display_methods = [m.replace("_", " ").title() for m in methods]

        # Build F1 matrix
        f1_matrix = np.zeros((len(defect_types), len(methods)))
        for i, dt in enumerate(defect_types):
            for j, m in enumerate(methods):
                if m in type_summary[dt]:
                    f1_matrix[i, j] = type_summary[dt][m]["f1"]

        fig, ax = plt.subplots(figsize=(10, max(4, len(defect_types) * 0.8)))

        im = ax.imshow(f1_matrix, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)

        ax.set_xticks(np.arange(len(methods)))
        ax.set_yticks(np.arange(len(defect_types)))
        ax.set_xticklabels(display_methods, fontsize=10, fontweight="bold")
        ax.set_yticklabels(defect_types, fontsize=10)

        # Annotate cells with F1 values
        for i in range(len(defect_types)):
            for j in range(len(methods)):
                val = f1_matrix[i, j]
                text_color = "white" if val < 0.4 else "black"
                ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                        fontsize=10, fontweight="bold", color=text_color)

        # Add count column
        for i, dt in enumerate(defect_types):
            if dt in type_summary and methods:
                first_method = list(type_summary[dt].values())[0]
                count = first_method.get("count", 0)
                ax.text(len(methods) + 0.1, i, f"n={count}", va="center",
                        fontsize=9, color=COLORS["gray"])

        ax.set_title("F1 Score — Method × Defect Type",
                     fontsize=13, fontweight="bold", pad=15)

        cbar = plt.colorbar(im, ax=ax, fraction=0.03, pad=0.04)
        cbar.set_label("F1 Score", fontsize=11)

        plt.tight_layout()
        return self._save_figure(fig, "defect_type_heatmap")

    # ── Example Gallery ──

    def add_example_gallery(self, example_results: list) -> str:
        """Grid showing one example per defect type: original | restored | mask GT | detection."""
        n = len(example_results)
        if n == 0:
            return ""

        fig, axes = plt.subplots(n, 4, figsize=(20, 3.5 * n))
        if n == 1:
            axes = axes.reshape(1, -1)

        col_titles = ["Original", "Restored", "Ground Truth", "Detection (Edge)"]
        for j, title in enumerate(col_titles):
            axes[0, j].set_title(title, fontsize=12, fontweight="bold", pad=10)

        for i, ex in enumerate(example_results):
            # Original
            orig_rgb = cv2.cvtColor(ex["original"], cv2.COLOR_BGR2RGB)
            axes[i, 0].imshow(orig_rgb)
            axes[i, 0].set_ylabel(ex["defect_type"], fontsize=10, fontweight="bold", rotation=0,
                                   labelpad=80, va="center")

            # Restored
            rest_rgb = cv2.cvtColor(ex["restored"], cv2.COLOR_BGR2RGB)
            axes[i, 1].imshow(rest_rgb)
            axes[i, 1].text(5, 20, f"method: {ex['best_method']}",
                            fontsize=8, color="white",
                            bbox=dict(boxstyle="round,pad=0.3", facecolor="black", alpha=0.7))

            # Ground truth mask
            axes[i, 2].imshow(ex["mask_gt"], cmap="gray")

            # Detection result (edge annotated)
            if "edge_annotated" in ex["detection"]:
                det_rgb = cv2.cvtColor(ex["detection"]["edge_annotated"], cv2.COLOR_BGR2RGB)
                axes[i, 3].imshow(det_rgb)
            else:
                axes[i, 3].imshow(np.zeros_like(orig_rgb))

            for j in range(4):
                axes[i, j].set_xticks([])
                axes[i, j].set_yticks([])

        fig.suptitle("Detection Examples — One per Defect Type",
                     fontsize=15, fontweight="bold", y=1.01)
        plt.tight_layout()
        return self._save_figure(fig, "example_gallery")

    # ── Timing Summary ──

    def add_timing_summary(self, all_timings: list) -> str:
        """Histogram of processing times + stats box."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Histogram
        ax1.hist(all_timings, bins=20, color=COLORS["primary"], edgecolor="white",
                 linewidth=0.5, alpha=0.8)
        avg_ms = np.mean(all_timings)
        ax1.axvline(avg_ms, color=COLORS["accent"], linewidth=2, linestyle="--",
                    label=f"Mean: {avg_ms:.0f} ms")
        ax1.set_xlabel("Processing Time per Image (ms)", fontsize=11)
        ax1.set_ylabel("Count", fontsize=11)
        ax1.set_title("Processing Time Distribution", fontsize=13, fontweight="bold")
        ax1.legend(fontsize=10)
        ax1.spines["top"].set_visible(False)
        ax1.spines["right"].set_visible(False)

        # Stats box
        stats = {
            "Mean": f"{np.mean(all_timings):.0f} ms",
            "Std": f"{np.std(all_timings):.0f} ms",
            "Min": f"{np.min(all_timings):.0f} ms",
            "Max": f"{np.max(all_timings):.0f} ms",
            "Median": f"{np.median(all_timings):.0f} ms",
            "FPS (avg)": f"{1000 / np.mean(all_timings):.1f}",
            "Images": f"{len(all_timings)}",
        }

        ax2.axis("off")
        table_data = [[k, v] for k, v in stats.items()]
        table = ax2.table(cellText=table_data, colLabels=["Metric", "Value"],
                          cellLoc="center", loc="center",
                          colColours=[COLORS["primary"], COLORS["primary"]])
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1, 2)

        # Style header
        for key, cell in table.get_celld().items():
            if key[0] == 0:
                cell.set_text_props(color="white", fontweight="bold")
            cell.set_edgecolor("white")
            cell.set_linewidth(2)

        ax2.set_title("Performance Statistics", fontsize=13, fontweight="bold", pad=20)

        fig.suptitle("Phase 3 — Timing Analysis", fontsize=15, fontweight="bold", y=1.02)
        plt.tight_layout()
        return self._save_figure(fig, "timing_summary")

    # ── Feasibility Summary ──

    def add_feasibility_summary(self, feasibility: dict) -> str:
        """Visual feasibility indicator."""
        fig, ax = plt.subplots(figsize=(8, 5))

        categories = ["Measured FPS", "Target FPS", "Max Speed\n(m/min)"]
        values = [
            feasibility["measured_fps"],
            feasibility["target_fps"],
            feasibility["max_conveyor_speed_m_min"],
        ]
        colors = [
            COLORS["primary"],
            COLORS["accent"],
            COLORS["secondary"] if feasibility["meets_requirement"] else COLORS["accent"],
        ]
        bars = ax.bar(categories, values, color=colors, width=0.5,
                      edgecolor="white", linewidth=1.5)

        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                    f"{val:.1f}", ha="center", fontweight="bold", fontsize=12)

        status = "VIABLE" if feasibility["meets_requirement"] else "NOT VIABLE"
        status_color = COLORS["secondary"] if feasibility["meets_requirement"] else COLORS["accent"]
        ax.set_title(f"Industrial Feasibility — {status}",
                     fontsize=14, fontweight="bold", color=status_color)
        ax.set_ylabel("Value")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        plt.tight_layout()
        return self._save_figure(fig, "feasibility")

    # ── Legacy: timing chart for single-image mode ──

    def add_timing_chart(
        self, timing_results: List[TimingResult], title: str = "Pipeline Performance"
    ) -> str:
        names = [r.name for r in timing_results]
        times_ms = [r.avg_ms for r in timing_results]

        fig, ax = plt.subplots(figsize=(10, 6))
        colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(names)))
        bars = ax.barh(names, times_ms, color=colors, edgecolor="white")
        ax.set_xlabel("Time (ms)")
        ax.set_title(title, fontsize=13, fontweight="bold")
        for bar, val in zip(bars, times_ms):
            ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                     f"{val:.1f}ms", va="center", fontsize=9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        plt.tight_layout()
        return self._save_figure(fig, "timing_chart")

    # ── Legacy: restoration comparison ──

    def add_restoration_comparison(self, images: dict, metrics: dict) -> str:
        n = len(images)
        fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
        if n == 1:
            axes = [axes]
        for ax, (name, img) in zip(axes, images.items()):
            display = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if len(img.shape) == 3 else img
            ax.imshow(display, cmap="gray" if len(img.shape) == 2 else None)
            title = name
            if name in metrics:
                m = metrics[name]
                if "psnr_restored" in m:
                    title += f"\nPSNR: {m['psnr_restored']:.1f}dB\nSSIM: {m['ssim_restored']:.3f}"
            ax.set_title(title, fontsize=10)
            ax.axis("off")
        plt.tight_layout()
        return self._save_figure(fig, "restoration_comparison")

    # ── Legacy: detection results ──

    def add_detection_results(self, annotated_image, detection_metrics: dict) -> str:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        display = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
        ax1.imshow(display)
        ax1.set_title("Detected Defects")
        ax1.axis("off")
        metric_names = list(detection_metrics.keys())
        metric_values = list(detection_metrics.values())
        colors = [COLORS["secondary"] if v > 0.7 else COLORS["accent"] for v in metric_values]
        ax2.barh(metric_names, metric_values, color=colors)
        ax2.set_xlim(0, 1)
        ax2.set_title("Detection Metrics")
        for i, v in enumerate(metric_values):
            ax2.text(v + 0.02, i, f"{v:.3f}", va="center")
        plt.tight_layout()
        return self._save_figure(fig, "detection_results")

    # ── HTML Generation ──

    def generate_html(
        self,
        title: str = "Textile Inspection Benchmark Report",
        extra_data: Optional[dict] = None,
    ) -> Path:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        image_tags = ""
        for img_path in self.images_saved:
            label = img_path.stem.replace("_", " ").title()
            image_tags += f'''
            <div class="chart">
                <img src="{img_path.name}" alt="{img_path.stem}" loading="lazy">
                <p>{label}</p>
            </div>
'''

        extra_section = ""
        if extra_data:
            extra_json = json.dumps(extra_data, indent=2, default=str, ensure_ascii=False)
            extra_section = f"""
            <h2>Raw Data (JSON)</h2>
            <details>
                <summary>Click to expand raw metrics data</summary>
                <pre>{extra_json}</pre>
            </details>"""

        html = f"""<!DOCTYPE html>
<html lang="pt-BR">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{title}</title>
<style>
    * {{ margin: 0; padding: 0; box-sizing: border-box; }}
    body {{
        font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, sans-serif;
        max-width: 1200px;
        margin: 0 auto;
        padding: 30px 20px;
        background: #f8f9fa;
        color: #2c3e50;
    }}
    h1 {{
        font-size: 1.8em;
        color: #2c3e50;
        border-bottom: 3px solid #3498db;
        padding-bottom: 12px;
        margin-bottom: 8px;
    }}
    h2 {{
        font-size: 1.3em;
        color: #34495e;
        margin: 30px 0 15px;
        padding-left: 10px;
        border-left: 4px solid #3498db;
    }}
    .meta {{
        color: #95a5a6;
        font-size: 0.9em;
        margin-bottom: 25px;
    }}
    .chart {{
        background: white;
        padding: 20px;
        margin: 20px 0;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        text-align: center;
        transition: transform 0.2s;
    }}
    .chart:hover {{
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.12);
    }}
    .chart img {{
        max-width: 100%;
        border-radius: 8px;
    }}
    .chart p {{
        color: #7f8c8d;
        font-style: italic;
        margin-top: 10px;
        font-size: 0.95em;
    }}
    details {{
        background: white;
        padding: 15px 20px;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        margin: 15px 0;
    }}
    summary {{
        cursor: pointer;
        font-weight: 600;
        color: #3498db;
        padding: 5px 0;
    }}
    pre {{
        background: #2c3e50;
        color: #ecf0f1;
        padding: 20px;
        border-radius: 8px;
        overflow-x: auto;
        font-size: 0.85em;
        line-height: 1.5;
        margin-top: 10px;
    }}
    footer {{
        text-align: center;
        color: #bdc3c7;
        font-size: 0.8em;
        margin-top: 40px;
        padding-top: 20px;
        border-top: 1px solid #ecf0f1;
    }}
</style>
</head>
<body>
<h1>{title}</h1>
<p class="meta">Generated: {timestamp} | UFSC — Computer Vision Project</p>
{image_tags}
{extra_section}
<footer>
    Textile Inspection System &mdash; UFSC Computer Vision Project &mdash; {timestamp}
</footer>
</body>
</html>"""

        report_path = self.output_dir / "report.html"
        report_path.write_text(html, encoding="utf-8")
        print(f"  Report generated: {report_path}")
        return report_path
