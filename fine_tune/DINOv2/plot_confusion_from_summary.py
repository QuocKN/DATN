from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def safe_div(num: float, den: float) -> float:
    return num / den if den else 0.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot confusion matrix image from a train_dinov2_binary summary.json file."
    )
    parser.add_argument(
        "--summary",
        type=str,
        required=True,
        help="Path to summary.json produced by train_dinov2_binary.py",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="",
        help="Optional output image path (default: same folder as summary, filename confusion_matrix.png)",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="Confusion Matrix",
        help="Figure title",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary_path = Path(args.summary)
    if not summary_path.exists():
        raise FileNotFoundError(f"Summary file not found: {summary_path}")

    with summary_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    cm = np.array(data.get("confusion_matrix", []), dtype=np.int64)
    class_names = data.get("class_names", [str(i) for i in range(cm.shape[0])])

    if cm.ndim != 2 or cm.shape[0] == 0 or cm.shape[0] != cm.shape[1]:
        raise ValueError("Invalid or missing confusion_matrix in summary.json.")

    out_path = Path(args.out) if args.out else (summary_path.parent / "confusion_matrix.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")

    ax.set_title(args.title)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=30, ha="right")
    ax.set_yticklabels(class_names)

    threshold = cm.max() / 2.0 if cm.size else 0.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = "white" if cm[i, j] > threshold else "black"
            ax.text(j, i, f"{cm[i, j]}", ha="center", va="center", color=color, fontsize=11)

    tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])
    accuracy = safe_div(tn + tp, tn + fp + fn + tp)
    precision_drone = safe_div(tp, tp + fp)
    recall_drone = safe_div(tp, tp + fn)
    metrics_text = (
        f"Accuracy: {accuracy * 100:.2f}%   |   "
        f"Precision (drone): {precision_drone * 100:.2f}%   |   "
        f"Recall (drone): {recall_drone * 100:.2f}%"
    )
    fig.text(0.5, 0.01, metrics_text, ha="center", va="bottom", fontsize=10)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout(rect=[0.0, 0.05, 1.0, 1.0])
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

    print(f"Saved confusion matrix image: {out_path}")


if __name__ == "__main__":
    main()
