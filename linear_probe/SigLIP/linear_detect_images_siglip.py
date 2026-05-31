from __future__ import annotations

import json
from pathlib import Path
from typing import List, Sequence

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

try:
    from transformers import AutoImageProcessor, SiglipVisionModel
except Exception as exc:  # pragma: no cover
    raise RuntimeError(
        "Missing dependency 'transformers'. Install it first: pip install transformers"
    ) from exc

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tif", ".tiff"}
FEATURE_EXTRACTOR = "siglip_base_patch16_224"

# Edit these values directly.
ARTIFACT_IN = "linear_probe/SigLIP/trained_classifier.joblib"
SOURCE_DIR = "/home/quocnk/Documents/NKQuoc/Data/RF/Tu_thu/drone2/spectrograms"
OUTPUT_JSON = "linear_probe/SigLIP/report/Tu_thu/2toan/results.json"
OUTPUT_CHART = "linear_probe/SigLIP/report/Tu_thu/2toan/results_chart.png"
SIGLIP_MODEL = "google/siglip-base-patch16-224"
DEVICE = "cuda:0"
BATCH_SIZE = 64


def collect_image_paths(root: Path) -> List[Path]:
    if not root.exists():
        raise FileNotFoundError(f"Path does not exist: {root}")
    return sorted([p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS])


def load_feature_model(device: torch.device, model_name: str):
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = SiglipVisionModel.from_pretrained(model_name)
    model.eval()
    model.to(device)
    return model, processor


def extract_embeddings(
    image_paths: Sequence[Path],
    model,
    processor,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    embs: List[np.ndarray] = []

    with torch.no_grad():
        for start in tqdm(range(0, len(image_paths), batch_size), desc="Extract embeddings", unit="batch"):
            batch_paths = image_paths[start : start + batch_size]
            images = [Image.open(path).convert("RGB") for path in batch_paths]
            inputs = processor(images=images, return_tensors="pt")
            pixel_values = inputs["pixel_values"].to(device, non_blocking=True)

            outputs = model(pixel_values=pixel_values)
            features = outputs.pooler_output.detach().cpu().numpy()
            embs.append(features)

    if not embs:
        raise ValueError("No image found for embedding extraction")

    return np.concatenate(embs, axis=0)


def visualize_detection_results(
    scores: np.ndarray,
    pred: np.ndarray,
    output_path: Path,
    drone_count: int,
    total: int,
    model_name: str | None,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.patch.set_facecolor("#f7f9fc")

    drone_mask = pred == 1
    non_drone_mask = pred == 0

    ax1 = axes[0, 0]
    if np.any(drone_mask):
        ax1.hist(scores[drone_mask], bins=30, alpha=0.75, color="#2b8a6e", edgecolor="black", label="Predicted drone")
    if np.any(non_drone_mask):
        ax1.hist(scores[non_drone_mask], bins=30, alpha=0.75, color="#c94949", edgecolor="black", label="Predicted non-drone")
    ax1.axvline(0.5, color="#111827", linestyle="--", linewidth=2, label="Decision threshold: 0.5")
    ax1.set_xlabel("Drone score")
    ax1.set_ylabel("Number of images")
    ax1.set_title("Score Distribution", fontweight="bold")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    ax2 = axes[0, 1]
    indices = np.arange(total)
    if np.any(drone_mask):
        ax2.scatter(indices[drone_mask], scores[drone_mask], s=18, color="#2b8a6e", alpha=0.75, label="Predicted drone")
    if np.any(non_drone_mask):
        ax2.scatter(indices[non_drone_mask], scores[non_drone_mask], s=18, color="#c94949", alpha=0.75, label="Predicted non-drone")
    ax2.axhline(0.5, color="#111827", linestyle="--", linewidth=2)
    ax2.set_xlabel("Image index")
    ax2.set_ylabel("Drone score")
    ax2.set_title("Score by Image Order", fontweight="bold")
    ax2.set_ylim(-0.03, 1.03)
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    ax3 = axes[1, 0]
    counts = [drone_count, total - drone_count]
    labels = ["Drone", "Non-drone"]
    colors = ["#2b8a6e", "#c94949"]
    ax3.pie(
        counts,
        labels=labels,
        colors=colors,
        autopct=lambda pct: f"{pct:.1f}%\n({int(round(pct / 100 * total))})",
        startangle=90,
        textprops={"fontsize": 11},
    )
    ax3.set_title("Prediction Split", fontweight="bold")

    ax4 = axes[1, 1]
    metric_names = ["Source images", "Drone", "Non-drone", "Drone rate"]
    metric_values = [total, drone_count, total - drone_count, drone_count / max(total, 1)]
    display_values = [str(total), str(drone_count), str(total - drone_count), f"{100 * metric_values[-1]:.1f}%"]
    bar_values = [total, drone_count, total - drone_count, total * metric_values[-1]]
    bars = ax4.bar(metric_names, bar_values, color=["#357ABD", "#2b8a6e", "#c94949", "#d9902f"])
    ax4.set_title("Detection Summary", fontweight="bold")
    ax4.set_ylabel("Count-scaled value")
    ax4.grid(axis="y", alpha=0.3)
    for bar, text in zip(bars, display_values):
        ax4.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(total * 0.015, 0.5), text, ha="center", fontweight="bold")

    fig.suptitle(
        f"Linear Probe Detection Results | Model: {model_name or 'unknown'} | "
        f"Drone: {drone_count}/{total} ({100 * drone_count / max(total, 1):.1f}%)",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout(rect=(0, 0, 1, 0.95))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved chart: {output_path}")


def main() -> None:
    artifact_in = Path(ARTIFACT_IN)
    source_root = Path(SOURCE_DIR)

    if not artifact_in.exists():
        raise FileNotFoundError(f"Artifact file not found: {artifact_in}")

    payload = joblib.load(artifact_in)
    if payload.get("artifact_type") != "linear_probe_classifier":
        raise ValueError("This script expects a linear_probe_classifier artifact")

    clf = payload["model"]
    summary_train = payload.get("summary", {})
    feature_extractor = summary_train.get("feature_extractor")
    if feature_extractor and feature_extractor != FEATURE_EXTRACTOR:
        raise ValueError(
            "This detector expects a SigLIP artifact, "
            f"but artifact feature_extractor={feature_extractor!r}"
        )

    device = torch.device(DEVICE if torch.cuda.is_available() and DEVICE.startswith("cuda") else "cpu")

    source_paths = collect_image_paths(source_root)
    model_name = summary_train.get("siglip_model") or SIGLIP_MODEL
    model, processor = load_feature_model(device, model_name)
    embeddings = extract_embeddings(
        source_paths,
        model,
        processor,
        device,
        batch_size=BATCH_SIZE,
    )

    pred = clf.predict(embeddings)
    pred = pred.astype(int)

    if hasattr(clf, "predict_proba"):
        scores = clf.predict_proba(embeddings)[:, 1]
    elif hasattr(clf, "decision_function"):
        raw = clf.decision_function(embeddings)
        scores = 1.0 / (1.0 + np.exp(-raw))
    else:
        scores = pred.astype(np.float32)

    predictions = []
    for path, y_hat, score in zip(source_paths, pred, scores):
        predictions.append(
            {
                "image": str(path),
                "prediction": "drone" if int(y_hat) == 1 else "non_drone",
                "drone_score": float(score),
            }
        )

    drone_count = int(np.sum(pred == 1))
    total = int(len(pred))

    summary = {
        "method": "linear_probe_classifier",
        "feature_extractor": FEATURE_EXTRACTOR,
        "siglip_model": model_name,
        "model_name": payload.get("model_name"),
        "artifact_in": str(artifact_in),
        "source_images": total,
        "detected_drone_count": drone_count,
        "detected_non_drone_count": total - drone_count,
        "detected_drone_rate": float(drone_count / max(total, 1)),
        "score_mean": float(np.mean(scores)) if total else None,
        "score_std": float(np.std(scores)) if total else None,
        "train_summary": summary_train,
    }

    output_path = Path(OUTPUT_JSON)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump({"summary": summary, "predictions": predictions}, f, indent=2, ensure_ascii=True)

    print(json.dumps(summary, indent=2, ensure_ascii=True))
    print(f"Saved JSON: {output_path}")

    chart_path = Path(OUTPUT_CHART)
    visualize_detection_results(
        scores=scores,
        pred=pred,
        output_path=chart_path,
        drone_count=drone_count,
        total=total,
        model_name=payload.get("model_name"),
    )


if __name__ == "__main__":
    main()
