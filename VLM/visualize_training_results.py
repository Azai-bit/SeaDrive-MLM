#!/usr/bin/env python3
"""Visualize Qwen3-VL fine-tuning artifacts from a training_results directory."""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_RESULTS_DIR = SCRIPT_DIR / "training_results"


def as_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        parsed = float(value)
    except Exception:
        return None
    return parsed if math.isfinite(parsed) else None


def load_json(path: Path, default: Any) -> Any:
    if not path.is_file():
        return default
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_log_history(results_dir: Path) -> list[dict[str, Any]]:
    json_path = results_dir / "log_history.json"
    if json_path.is_file():
        data = load_json(json_path, [])
        if isinstance(data, list):
            return [row for row in data if isinstance(row, dict)]

    csv_path = results_dir / "log_history.csv"
    if not csv_path.is_file():
        return []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        rows: list[dict[str, Any]] = []
        for row in csv.DictReader(f):
            rows.append({key: as_float(value) for key, value in row.items()})
    return rows


def series(log_history: list[dict[str, Any]], x_key: str, y_key: str) -> tuple[list[float], list[float]]:
    points: list[tuple[float, float]] = []
    for row in log_history:
        x = as_float(row.get(x_key))
        y = as_float(row.get(y_key))
        if x is not None and y is not None:
            points.append((x, y))
    points.sort(key=lambda item: item[0])
    if not points:
        return [], []
    xs, ys = zip(*points)
    return list(xs), list(ys)


def set_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.grid": True,
            "grid.alpha": 0.28,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.size": 11,
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "legend.fontsize": 10,
        }
    )


def save_figure(fig, path: Path, dpi: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_loss(log_history: list[dict[str, Any]], out_dir: Path, dpi: int) -> Path | None:
    train_x, train_y = series(log_history, "step", "loss")
    eval_x, eval_y = series(log_history, "step", "eval_loss")
    if not train_x and not eval_x:
        return None

    fig, ax = plt.subplots(figsize=(9, 5))
    if train_x:
        ax.plot(train_x, train_y, marker="o", markersize=3.5, linewidth=1.8, label="train loss")
    if eval_x:
        ax.plot(eval_x, eval_y, marker="s", markersize=4.0, linewidth=1.8, label="eval loss")
    ax.set_title("Loss")
    ax.set_xlabel("step")
    ax.set_ylabel("loss")
    ax.legend()
    path = out_dir / "loss.png"
    save_figure(fig, path, dpi)
    return path


def plot_learning_rate(log_history: list[dict[str, Any]], out_dir: Path, dpi: int) -> Path | None:
    xs, ys = series(log_history, "step", "learning_rate")
    if not xs:
        return None

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(xs, ys, marker="o", markersize=3.5, linewidth=1.8, color="#2f6f8f")
    ax.set_title("Learning Rate Schedule")
    ax.set_xlabel("step")
    ax.set_ylabel("learning rate")
    path = out_dir / "learning_rate.png"
    save_figure(fig, path, dpi)
    return path


def plot_grad_norm(log_history: list[dict[str, Any]], out_dir: Path, dpi: int) -> Path | None:
    xs, ys = series(log_history, "step", "grad_norm")
    if not xs:
        return None

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(xs, ys, marker="o", markersize=3.5, linewidth=1.8, color="#8f4f2f")
    ax.set_title("Gradient Norm")
    ax.set_xlabel("step")
    ax.set_ylabel("grad norm")
    path = out_dir / "grad_norm.png"
    save_figure(fig, path, dpi)
    return path


def plot_metric_summary(
    train_metrics: dict[str, Any],
    eval_metrics: dict[str, Any],
    out_dir: Path,
    dpi: int,
) -> Path | None:
    metric_groups = [
        ("Loss", [("train", "train_loss", train_metrics), ("eval", "eval_loss", eval_metrics)]),
        (
            "Runtime (min)",
            [
                ("train", "train_runtime", train_metrics),
                ("eval", "eval_runtime", eval_metrics),
            ],
        ),
        (
            "Samples / second",
            [
                ("train", "train_samples_per_second", train_metrics),
                ("eval", "eval_samples_per_second", eval_metrics),
            ],
        ),
        (
            "Steps / second",
            [
                ("train", "train_steps_per_second", train_metrics),
                ("eval", "eval_steps_per_second", eval_metrics),
            ],
        ),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    drew_any = False
    for ax, (title, specs) in zip(axes.flat, metric_groups):
        labels: list[str] = []
        values: list[float] = []
        for label, key, source in specs:
            value = as_float(source.get(key))
            if value is None:
                continue
            if key.endswith("_runtime"):
                value /= 60.0
            labels.append(label)
            values.append(value)
        if not values:
            ax.axis("off")
            continue
        drew_any = True
        bars = ax.bar(labels, values, color=["#3b82f6", "#10b981"][: len(values)])
        ax.set_title(title)
        for bar, value in zip(bars, values):
            ax.annotate(
                f"{value:.3g}",
                (bar.get_x() + bar.get_width() / 2.0, bar.get_height()),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=10,
            )
    if not drew_any:
        plt.close(fig)
        return None
    fig.suptitle("Metric Summary", fontsize=16)
    path = out_dir / "metric_summary.png"
    save_figure(fig, path, dpi)
    return path


def plot_dashboard(
    log_history: list[dict[str, Any]],
    train_metrics: dict[str, Any],
    eval_metrics: dict[str, Any],
    run_config: dict[str, Any],
    out_dir: Path,
    dpi: int,
) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.8), constrained_layout=True)
    fig.set_constrained_layout_pads(w_pad=0.08, h_pad=0.12, hspace=0.22, wspace=0.10)

    ax = axes[0][0]
    train_x, train_y = series(log_history, "step", "loss")
    eval_x, eval_y = series(log_history, "step", "eval_loss")
    if train_x:
        ax.plot(train_x, train_y, marker="o", markersize=3, linewidth=1.6, label="train loss")
    if eval_x:
        ax.plot(eval_x, eval_y, marker="s", markersize=4, linewidth=1.6, label="eval loss")
    ax.set_title("Loss")
    ax.set_xlabel("step")
    ax.set_ylabel("loss")
    ax.legend()

    ax = axes[0][1]
    lr_x, lr_y = series(log_history, "step", "learning_rate")
    if lr_x:
        ax.plot(lr_x, lr_y, marker="o", markersize=3, linewidth=1.6, color="#2f6f8f")
    ax.set_title("Learning Rate")
    ax.set_xlabel("step")
    ax.set_ylabel("learning rate")

    ax = axes[1][0]
    grad_x, grad_y = series(log_history, "step", "grad_norm")
    if grad_x:
        ax.plot(grad_x, grad_y, marker="o", markersize=3, linewidth=1.6, color="#8f4f2f")
    ax.set_title("Gradient Norm")
    ax.set_xlabel("step")
    ax.set_ylabel("grad norm")

    ax = axes[1][1]
    ax.axis("off")
    summary_lines = [
        f"train_loss: {format_metric(train_metrics.get('train_loss'))}",
        f"eval_loss: {format_metric(eval_metrics.get('eval_loss'))}",
        f"epoch: {format_metric(train_metrics.get('epoch'))}",
        f"train_samples: {format_metric(train_metrics.get('train_samples'), digits=0)}",
        f"eval_samples: {format_metric(eval_metrics.get('eval_samples'), digits=0)}",
        f"runtime_min: {format_metric(runtime_minutes(train_metrics.get('train_runtime')))}",
        f"learning_rate: {format_metric(run_config.get('learning_rate'))}",
        f"batch/accum: {run_config.get('per_device_train_batch_size', 'N/A')}/"
        f"{run_config.get('gradient_accumulation_steps', 'N/A')}",
        f"LoRA r/alpha: {run_config.get('lora_r', 'N/A')}/"
        f"{run_config.get('lora_alpha', 'N/A')}",
    ]
    ax.text(
        0.02,
        0.98,
        "\n".join(summary_lines),
        ha="left",
        va="top",
        family="monospace",
        fontsize=11,
        linespacing=1.5,
    )
    ax.set_title("Run Summary", loc="left")

    fig.suptitle("Qwen3-VL COLREG Fine-tuning", fontsize=17, y=1.02)
    path = out_dir / "dashboard.png"
    save_figure(fig, path, dpi)
    return path


def runtime_minutes(value: Any) -> float | None:
    parsed = as_float(value)
    if parsed is None:
        return None
    return parsed / 60.0


def format_metric(value: Any, digits: int = 4) -> str:
    parsed = as_float(value)
    if parsed is None:
        return "N/A"
    if digits <= 0:
        return str(int(round(parsed)))
    return f"{parsed:.{digits}g}"


def write_report(
    out_dir: Path,
    results_dir: Path,
    train_metrics: dict[str, Any],
    eval_metrics: dict[str, Any],
    run_config: dict[str, Any],
    generated: list[Path],
) -> Path:
    lines = [
        "Training Results Visualization",
        "=" * 30,
        f"results_dir: {results_dir.resolve()}",
        f"output_dir: {out_dir.resolve()}",
        "",
        "Metrics",
        f"- train_loss: {format_metric(train_metrics.get('train_loss'))}",
        f"- eval_loss: {format_metric(eval_metrics.get('eval_loss'))}",
        f"- epoch: {format_metric(train_metrics.get('epoch'))}",
        f"- train_runtime_min: {format_metric(runtime_minutes(train_metrics.get('train_runtime')))}",
        f"- train_samples: {format_metric(train_metrics.get('train_samples'), digits=0)}",
        f"- eval_samples: {format_metric(eval_metrics.get('eval_samples'), digits=0)}",
        "",
        "Run Config",
        f"- model_path: {run_config.get('model_path', 'N/A')}",
        f"- dataset_dir: {run_config.get('dataset_dir', 'N/A')}",
        f"- output_model_path: {run_config.get('output_model_path', 'N/A')}",
        f"- learning_rate: {run_config.get('learning_rate', 'N/A')}",
        f"- num_train_epochs: {run_config.get('num_train_epochs', 'N/A')}",
        f"- batch/accum: {run_config.get('per_device_train_batch_size', 'N/A')}/"
        f"{run_config.get('gradient_accumulation_steps', 'N/A')}",
        f"- LoRA r/alpha/dropout: {run_config.get('lora_r', 'N/A')}/"
        f"{run_config.get('lora_alpha', 'N/A')}/{run_config.get('lora_dropout', 'N/A')}",
        "",
        "Generated Figures",
    ]
    lines.extend(f"- {path.name}" for path in generated)

    report_path = out_dir / "visualization_report.txt"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def write_html_index(out_dir: Path, generated: list[Path], report_path: Path) -> Path:
    items = []
    for path in generated:
        rel = path.name
        title = path.stem.replace("_", " ").title()
        items.append(
            "<section>"
            f"<h2>{html.escape(title)}</h2>"
            f"<img src=\"{html.escape(rel)}\" alt=\"{html.escape(title)}\">"
            "</section>"
        )
    body = "\n".join(items)
    index = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Training Results Visualization</title>
  <style>
    body {{ margin: 24px; font-family: Arial, sans-serif; color: #111827; }}
    h1 {{ margin-bottom: 4px; }}
    a {{ color: #2563eb; }}
    section {{ margin-top: 28px; }}
    img {{ max-width: 100%; border: 1px solid #d1d5db; }}
  </style>
</head>
<body>
  <h1>Training Results Visualization</h1>
  <p><a href="{html.escape(report_path.name)}">Text report</a></p>
  {body}
</body>
</html>
"""
    index_path = out_dir / "index.html"
    index_path.write_text(index, encoding="utf-8")
    return index_path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Visualize train_metrics/eval_metrics/log_history from training_results."
    )
    parser.add_argument(
        "results_dir",
        nargs="?",
        default=str(DEFAULT_RESULTS_DIR),
        help="training_results directory. Default: xtf/LLM/training_results",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="output directory. Default: <results_dir>/visualization",
    )
    parser.add_argument("--dpi", type=int, default=160, help="PNG resolution")
    parser.add_argument("--no-html", action="store_true", help="do not generate index.html")
    args = parser.parse_args()

    results_dir = Path(args.results_dir).expanduser().resolve()
    if not results_dir.is_dir():
        raise FileNotFoundError(f"training_results directory not found: {results_dir}")

    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else results_dir / "visualization"
    out_dir.mkdir(parents=True, exist_ok=True)
    set_style()

    log_history = load_log_history(results_dir)
    train_metrics = load_json(results_dir / "train_metrics.json", {})
    eval_metrics = load_json(results_dir / "eval_metrics.json", {})
    run_config = load_json(results_dir / "run_config.json", {})

    generated: list[Path] = []
    for path in (
        plot_loss(log_history, out_dir, args.dpi),
        plot_learning_rate(log_history, out_dir, args.dpi),
        plot_grad_norm(log_history, out_dir, args.dpi),
        plot_metric_summary(train_metrics, eval_metrics, out_dir, args.dpi),
        plot_dashboard(log_history, train_metrics, eval_metrics, run_config, out_dir, args.dpi),
    ):
        if path is not None:
            generated.append(path)

    report_path = write_report(out_dir, results_dir, train_metrics, eval_metrics, run_config, generated)
    print(f"saved: {report_path}")
    if not args.no_html:
        index_path = write_html_index(out_dir, generated, report_path)
        print(f"saved: {index_path}")
    for path in generated:
        print(f"saved: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
